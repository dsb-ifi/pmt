"""Check reference cue injection, memory readout and EOS metrics."""


import torch
import pytest
from types import SimpleNamespace as NS

from pmt.config import load_run
from pmt.cue import memory_features
from pmt.model import Encoder


@pytest.mark.parametrize('kind', ('pma', 'spma', 'txl', 'xlstm'))
@torch.no_grad()
def test_cue_window_readout(kind: str) -> None:
    """Request window readouts for cue, preserving each backbone's default outputs."""
    if kind == 'xlstm':
        pytest.importorskip('xlstm')
    cfg = load_run(
        'table2_cue_UCR_FordA_pmt_main_s42', [f'pma.kind={kind}', 'pma.layers=1', 'model.dim=16', 'model.layers=1']
    )
    encoder = Encoder(cfg).eval()
    if kind == 'xlstm':
        encoder.pma.stack = torch.nn.Identity()
    x = torch.linspace(-1, 1, 500).view(1, 1, -1)
    ordinary, requested = encoder(x), encoder(x, return_all_states=True)
    assert torch.equal(requested.pma_final, ordinary.pma_final)
    if kind == 'xlstm':
        tokens = encoder.stem(x)[0][:, 1:]
        assert tokens.size(1) == 56 and requested.pma_states.size(1) == 5
        assert torch.equal(memory_features(requested.pma_states)[:, -1], tokens[:, 54])
        assert torch.equal(memory_features(ordinary.pma_states)[:, -1], tokens[:, 55])
    else:
        assert torch.equal(requested.pma_states, ordinary.pma_states)


class MemoryEncoder(torch.nn.Module):
    """Expose the fixed small encoder used by the cue replay checks."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, *, return_all_states: bool = False) -> object:
        """Summarize raw windows into two deterministic memory slots."""
        assert return_all_states
        windows = x.unfold(-1, 25, 13).squeeze(1)
        feats = torch.stack((windows.mean(-1), windows.amax(-1), windows.std(-1)), -1)
        return NS(pma_states=torch.stack((feats, feats * 0.7), 2))


