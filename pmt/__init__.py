"""Progressive Memory Transformer (PMT): minimal reference implementation."""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from pmt.model import Pretrained


def load_encoder(path: str | PathLike[str], device: str | torch.device = 'cpu') -> Pretrained:
    """Load a release encoder.pt or best_encoder.pt for evaluation.

    Pass raw channel-first series [B, C, T]; the module applies the TRAIN
    normalization saved with the encoder. It returns Features with cls (sequence
    summary), tokens, pma_states (memory states), pma_final, and mask.
    """
    import torch

    from pmt.config import Config
    from pmt.model import Pretrained, build_encoder

    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    encoder = build_encoder(Config(payload['cfg']))
    encoder.load_state_dict(payload['state_dict'], strict=True)
    return Pretrained(encoder, payload.get('mean'), payload.get('std')).to(device).eval()
