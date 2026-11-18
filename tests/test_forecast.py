"""Check reference forecast targets, ridge fit and checkpoint selection."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace as NS

import numpy as np
import torch
import pytest

from pmt.forecast import sliding_encode


class SumEncoder(torch.nn.Module):
    """Expose cumulative token features to make every padded history position observable."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> object:
        """Return window-prefix sums after replacing only padding NaNs."""
        return NS(tokens=torch.nan_to_num(x).transpose(1, 2).cumsum(1))


def test_sliding_readouts_own_only_the_last_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check compact storage for full and singleton batches without changing values."""
    from pmt import forecast

    batch_sizes = []

    def concatenate(arrays: tuple[np.ndarray, ...] | list[np.ndarray]) -> np.ndarray:
        if isinstance(arrays, list):
            for array in arrays:
                assert array.base.untyped_storage().nbytes() == array.nbytes
                batch_sizes.append(len(array))
        return np.concatenate(arrays)

    monkeypatch.setattr(forecast, 'np', NS(full=np.full, nan=np.nan, concatenate=concatenate))
    series = np.arange(10, dtype=np.float32).reshape(5, 2)
    np.testing.assert_array_equal(sliding_encode(SumEncoder(), series, batch_size=2), series.cumsum(axis=0))
    assert batch_sizes == [2, 2, 1]


def test_validation_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Load the selected retained milestone and reject a mismatched checkpoint epoch."""
    from pmt import forecast

    rows = [
        dict(epoch=e, step=e * 100, **{'forecast/uni/avg_val_mae_norm': mae, 'forecast/uni/avg_val_mse_norm': mse})
        for e, mae, mse in (
            (2, 0.3, 0.5),
            (6, 0.2, 0.4),
            (10, 0.2, 0.4),
            (20, 0.3, 0.4),
            (30, 0.3, 0.4),
            (40, 0.3, 0.4),
            (50, 0.3, 0.4),
        )
    ]
    history = tmp_path / 'metrics.json'
    history.write_text(json.dumps({'history': rows[:-1]}))
    with pytest.raises(ValueError, match=r'expected Table 3 validation milestones.*found \[2, 6, 10, 20, 30, 40\]'):
        forecast.replay(tmp_path, NS())
    history.write_text(json.dumps({'history': rows}))
    (tmp_path / 'checkpoints').mkdir()
    path = tmp_path / 'checkpoints/epoch6.pt'
    payload = dict(epoch=6, cfg={}, state_dict={'dummy': torch.tensor([2.0])})
    torch.save(payload, path)
    monkeypatch.setitem(sys.modules, 'pmt.model', NS(build_encoder=lambda cfg: SumEncoder()))
    monkeypatch.setattr(forecast, 'evaluate', lambda model, splits, cfg, **kw: (float(model.dummy.detach()), kw))
    result = forecast.replay(tmp_path, NS(), device='cpu')
    assert result['epoch'] == 6
    assert result['wide23'] == (2.0, {'test': True, 'alphas': forecast.WIDE23})
    assert result['source13'] == (2.0, {'test': True, 'alphas': forecast.SOURCE13})
    payload['epoch'] = 10
    torch.save(payload, path)
    with pytest.raises(ValueError, match='epoch disagrees'):
        forecast.replay(tmp_path, NS())
