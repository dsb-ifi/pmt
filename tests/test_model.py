"""Model construction, attention masking and gradient checks."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pmt.config import load_run
from pmt.data import load_dataset
from pmt.model import SSLModel
from pmt.pma import Attention, Overlap, sanitize

pytestmark = pytest.mark.unit


def test_table3_training_initialization(tmp_path: Path) -> None:
    """Match the reference Table 3 training entry point after data setup and construction."""
    folder = tmp_path / 'ETT-small'
    folder.mkdir()
    rows = ''.join(f'2020-01-{i // 24 + 1:02d} {i % 24:02d}:00:00,{i % 5},{i}\n' for i in range(48))
    (folder / 'ETTh1.csv').write_text('date,HUFL,OT\n' + rows)
    cfg = load_run('table3_pmt_uni_etth1_s43', [f'data.root={tmp_path}', 'data.length=4', 'data.horizons=[2]'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.train.seed)
        load_dataset(cfg)
        model = SSLModel(cfg)
        assert (
            hashlib.sha256(model.encoder.stem.cls_token.detach().numpy()).hexdigest()
            == '65bf42bdd92781be596fa687acee80713e4a823ead542906f5ecb3da6ea426e9'
        )
        assert (
            hashlib.sha256(torch.get_rng_state().numpy()).hexdigest()
            == '96a3e6a03b6954f570d76dbc8570192f996e1941caec5ca93c6bdbc33e47ff13'
        )


@pytest.mark.parametrize('dtype,negative', [(torch.float32, -1e9), (torch.bfloat16, -6e4), (torch.float16, -6e4)])
def test_attention_mask(dtype: torch.dtype, negative: float, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open fully blocked rows and pass finite sentinels in the projected Q dtype."""
    blocked = torch.ones((3, 3), dtype=torch.bool)
    expected = torch.tensor([[0, 0, negative], [0, 0, negative], [0, negative, 0]], dtype=dtype)

    def attention(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, attn_mask: torch.Tensor, dropout_p: float
    ) -> torch.Tensor:
        assert query.dtype == attn_mask.dtype == dtype
        torch.testing.assert_close(attn_mask, expected, rtol=0, atol=0)
        return query

    monkeypatch.setattr(torch.nn.functional, 'scaled_dot_product_attention', attention)
    layer = Attention(16).eval()
    with torch.autocast('cpu', dtype=dtype, enabled=dtype != torch.float32):
        layer(torch.zeros(1, 3, 16), sanitize(blocked))
    assert blocked.all()


@pytest.mark.parametrize(
    'kind,projector', [('pma', 'per_state'), ('spma', 'shared'), ('txl', 'pma'), ('xlstm', 'none')]
)
def test_training_backward(kind: str, projector: str) -> None:
    """Every backbone propagates finite gradients through both views and all heads."""
    if kind == 'xlstm':
        pytest.importorskip('xlstm')
    small = ['model.dim=16', 'model.layers=1', 'model.neighbors=2', 'pma.layers=2', 'pma.window=4', 'pma.stride=2']
    cfg = load_run(
        'pocconfirm_20261018_UCR_PhalangesOutlinesCorrect_confirmation_s42',
        small + [f'pma.kind={kind}', f'model.pcl_projector={projector}'],
    )
    model = SSLModel(cfg).train()
    if kind == 'txl':
        layer = model.encoder.pma.model.layers[0]
        assert 0.01 < float(layer.dec_attn.qkv_net.weight.detach().std()) < 0.03
        assert float(layer.dec_attn.layer_norm.weight.detach().std()) > 0.0
    x1, x2 = torch.randn(2, 1, 16, requires_grad=True), torch.randn(2, 1, 16, requires_grad=True)
    output = model(x1, x2)
    loss = output.proj_tokens_1.square().mean() + output.proj_cls_2.square().mean()
    loss = loss + output.proj_pma_1.square().mean() + output.pma_states_2.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    for x in (x1, x2):
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().max() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('rows,expected', [(65535, [65535]), (65536, [65535, 1])])
def test_overlap_attention_bound(monkeypatch: pytest.MonkeyPatch, rows: int, expected: list[int]) -> None:
    """Keep SDPA batches below the reference kernel limit without allocating attention scores."""
    calls = []

    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: object) -> torch.Tensor:
        calls.append(query.size(0))
        assert query.size(0) == key.size(0) == value.size(0) == kwargs['attn_mask'].size(0)
        return torch.zeros_like(query)

    monkeypatch.setattr(torch.nn.functional, 'scaled_dot_product_attention', attention)
    with torch.no_grad():
        Overlap(16, SimpleNamespace(conv_query=False, window=4, stride=2, gamma=1.0))(
            torch.ones(1, rows, 2, 16), torch.ones(1, rows, dtype=torch.bool)
        )
    assert calls == expected
