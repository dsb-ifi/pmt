"""PMT encoders and shared two-view projection heads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .pma import Attention, PMA, SkipGate, StateGate, norm, sanitize


@dataclass
class Features:
    """Encoder readouts and validity mask, retaining the PCL memory axes."""

    cls: Tensor
    tokens: Tensor
    pma_states: Tensor
    pma_final: Tensor
    mask: Tensor


@dataclass
class SSLOutputs:
    """Two views' projected features and the shared dedicated PCL projectors."""

    proj_cls_1: Tensor
    proj_cls_2: Tensor
    proj_tokens_1: Tensor
    proj_tokens_2: Tensor
    proj_pma_1: Tensor
    proj_pma_2: Tensor
    pma_states_1: Tensor
    pma_states_2: Tensor
    mask_1: Tensor
    mask_2: Tensor
    pcl_projector: nn.Module | None


def _cnn_stem(channels: int, cfg: Any) -> tuple[nn.Conv1d, nn.Module, nn.Dropout, nn.Linear, nn.Parameter]:
    """Build the standard stem in reference initialization order."""
    conv = nn.Conv1d(channels, 256, cfg.kernel, cfg.stride, cfg.kernel // 2, bias=False)
    if cfg.stem_norm == 'batchnorm':
        stem_norm = nn.BatchNorm1d(256, eps=1e-8)
    elif cfg.stem_norm == 'groupnorm':
        stem_norm = nn.GroupNorm(16, 256, eps=1e-8)
    else:
        raise ValueError(f'Unknown stem normalization {cfg.stem_norm}')
    dropout, projection = nn.Dropout(cfg.stem_dropout), nn.Linear(256, cfg.dim)
    cls = nn.Parameter(torch.randn(1, 1, 256) * 0.02)
    nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
    return conv, stem_norm, dropout, projection, cls


class Tokenizer(nn.Module):
    """Same-padded CNN or affine-patch tokenizer with an explicit CLS token."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg.model
        self.channels = cfg.data.channels
        if cfg.data.length < 1 or self.channels < 1:
            raise ValueError('Configured input channels and length must be positive')
        cfg = cfg.model
        if cfg.stem == 'linear_patch':
            with torch.random.fork_rng(devices=[]):
                self.patch_projection = nn.Conv1d(self.channels, cfg.dim, cfg.kernel, cfg.stride, cfg.kernel // 2)
                self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)
            # The reference isolates the affine stem's draws, then builds and discards the CNN stem.
            _cnn_stem(self.channels, cfg)
        elif cfg.stem == 'standard':
            self.conv1d, self.norm, self.dropout, self.projection, self.cls_token = _cnn_stem(self.channels, cfg)
        else:
            raise ValueError(f'Unknown tokenizer {cfg.stem}')

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Tokenize `[B,C,L]`, with majority-valid receptive-field masks."""
        if x.ndim != 3 or x.size(1) != self.channels:
            raise ValueError('Expected input shape [batch, configured channels, time]')
        valid = (~torch.isnan(x).any(1)).unsqueeze(1).float()
        arguments = (self.cfg.kernel, self.cfg.stride, self.cfg.kernel // 2)
        mask = (
            F.avg_pool1d(valid, *arguments) / F.avg_pool1d(torch.ones_like(valid), *arguments).clamp_min(1e-6)
        ).squeeze(1) > 0.5
        x = torch.nan_to_num(x, nan=0.0)
        if self.cfg.stem == 'standard':
            x = self.dropout(F.gelu(self.norm(self.conv1d(x))))
        else:
            x = self.patch_projection(x)
        tokens = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x.transpose(1, 2)), 1)
        mask = torch.cat((mask.new_ones(x.size(0), 1), mask), 1)
        if self.cfg.stem == 'standard':
            tokens = F.gelu(self.projection(tokens))
        return tokens, mask


class Position(nn.Module):
    """Add reference sinusoidal positions after the memory stack."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer('pe', self._make(2048, torch.device('cpu')))

    def _make(self, length: int, device: torch.device) -> Tensor:
        result = torch.zeros(length, self.dim, device=device)
        position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / self.dim)
        )
        result[:, 0::2], result[:, 1::2] = torch.sin(position * divisor), torch.cos(position * divisor)
        return result.unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        """Add positions, extending the deterministic cache for long sequences."""
        if x.size(1) > self.pe.size(1):
            self.pe = self._make(x.size(1), x.device)
        return x + self.pe[:, : x.size(1)].to(x.device)


class FeedForward(nn.Module):
    """GELU feedforward sublayer with reference parameter names."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear1, self.linear2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply two affine layers with GELU and both dropouts."""
        return self.dropout(self.linear2(self.dropout(F.gelu(self.linear1(x)))))


class NeighborhoodLayer(nn.Module):
    """Local causal self-attention with a global CLS readout row."""

    def __init__(self, dim: int, neighbors: int) -> None:
        super().__init__()
        self.self_attn, self.feed_forward = Attention(dim), FeedForward(dim)
        self.norm1, self.norm2, self.dropout = norm(dim), norm(dim), nn.Dropout(0.1)
        self.neighbors = neighbors

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        """Transform valid tokens with the historical mask openings."""
        length = x.size(1)
        query = torch.arange(length, device=x.device).unsqueeze(1)
        key = torch.arange(length, device=x.device).unsqueeze(0)
        blocked = ~((query == 0) | ((key <= query) & (key >= query - self.neighbors)))
        blocked = blocked.unsqueeze(0).unsqueeze(0)
        if not bool(valid.all()):
            invalid = ~valid.bool()
            row = invalid.unsqueeze(2).clone()
            row[:, 0] = False
            padding = row | invalid.unsqueeze(1)
            padding[..., 0] = False
            blocked = blocked | padding.unsqueeze(1)
            x = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
        attention = self.self_attn(self.norm1(x), sanitize(blocked))
        x = x + self.dropout(attention)
        return x + self.feed_forward(self.norm2(x))


class Neighborhood(nn.Module):
    """Neighborhood transformer with a terminal RMS normalization."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.layers = nn.ModuleList([NeighborhoodLayer(cfg.dim, cfg.neighbors) for _ in range(cfg.layers)])
        self.norm = norm(cfg.dim)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Apply every local transformer layer and the final normalization."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Encoder(nn.Module):
    """Tokenizer, memory stack and neighborhood encoder for one time series."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.position, self.stem, self.neighborhood = (Position(cfg.model.dim), Tokenizer(cfg), Neighborhood(cfg.model))
        if cfg.pma.kind in ('pma', 'spma'):
            self.pma = PMA(cfg.model.dim, cfg.pma)
        else:
            from .baselines import build_baseline

            self.pma = build_baseline(cfg)

    def _finish(self, tokens: Tensor, mask: Tensor, final: Tensor, states: Tensor) -> Features:
        encoded = self.neighborhood(self.position(tokens), mask)
        return Features(encoded[:, 0], encoded[:, 1:], states, final, mask)

    def _masked_stem(self, x: Tensor, mask_keep: float | None) -> tuple[Tensor, Tensor]:
        tokens, mask = self.stem(x)
        if mask_keep is not None:
            if not 0.0 <= mask_keep <= 1.0:
                raise ValueError('Token keep probability must lie in [0,1]')
            keep = torch.rand(tokens.size(0), tokens.size(1) - 1, device=x.device) < mask_keep
            keep = torch.cat((keep.new_ones(x.size(0), 1), keep), 1)
            tokens, mask = tokens.masked_fill(~keep.unsqueeze(-1), 0.0), mask & keep
        return tokens, mask

    def _one(self, x: Tensor, mask_keep: float | None, *, return_all_states: bool = False) -> Features:
        tokens, mask = self._masked_stem(x, mask_keep)
        if return_all_states and self.cfg.pma.kind == 'xlstm':
            output, final, states = self.pma(tokens[:, 1:], mask[:, 1:], return_all_states=True)
        else:
            output, final, states = self.pma(tokens[:, 1:], mask[:, 1:])
        return self._finish(torch.cat((tokens[:, :1], output), 1), mask, final, states)

    def forward(self, x: Tensor, *, return_all_states: bool = False) -> Features:
        """Encode without token masking, optionally keeping every memory window for cue probing."""
        return self._one(x, None, return_all_states=return_all_states)

    def two_views(self, x1: Tensor, x2: Tensor) -> tuple[Features, Features]:
        """Fuse memory and neighborhood work unless training masks are active."""
        keep = self.cfg.model.mask_keep if self.training else None
        if keep is not None:
            return self._one(x1, keep), self._one(x2, keep)
        tokens1, mask1 = self.stem(x1)
        tokens2, mask2 = self.stem(x2)
        output, final, states = self.pma(
            torch.cat((tokens1[:, 1:], tokens2[:, 1:]), 0), torch.cat((mask1[:, 1:], mask2[:, 1:]), 0)
        )
        batch = x1.size(0)
        out1, out2 = output.split((batch, x2.size(0)))
        one, two = torch.cat((tokens1[:, :1], out1), 1), torch.cat((tokens2[:, :1], out2), 1)
        encoded = self.neighborhood(
            torch.cat((self.position(one), self.position(two)), 0), torch.cat((mask1, mask2), 0)
        )
        encoded1, encoded2 = encoded.split((batch, x2.size(0)))
        return (
            Features(encoded1[:, 0], encoded1[:, 1:], states[:batch], final[:batch], mask1),
            Features(encoded2[:, 0], encoded2[:, 1:], states[batch:], final[batch:], mask2),
        )


class Projection(nn.Module):
    """Two-layer contrastive projector with a final LayerNorm."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 320))
        self.output_norm = nn.LayerNorm(320)
        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.trunc_normal_(self.layers[3].weight, std=0.02)
        nn.init.zeros_(self.layers[0].bias)
        nn.init.zeros_(self.layers[3].bias)

    def forward(self, x: Tensor) -> Tensor:
        """Project the final feature dimension and normalize it."""
        return self.output_norm(self.layers(x))


class SSLModel(nn.Module):
    """One shared encoder and shared projection heads for two augmented views."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.encoder = Encoder(cfg)
        dim = cfg.model.dim
        self.cls_head, self.token_head, self.pma_head = Projection(dim), Projection(dim), Projection(dim)
        topology = cfg.model.pcl_projector
        self._pcl_uses_pma = topology == 'pma'
        # The Table 2 recipe pairs the legacy skip gate with projector RNG restoration.
        restore_projector_rng = cfg.pma.skip_gate == 'tokenwise_legacy_constant_weight'
        with torch.random.fork_rng(devices=[], enabled=restore_projector_rng):
            if topology == 'per_state':
                self.pcl_projector = nn.ModuleList([Projection(dim) for _ in range(cfg.pma.states)])
            elif topology == 'shared':
                self.pcl_projector = Projection(dim)
                Projection(dim)  # The reference allocates a second dedicated head for view two.
            elif topology in ('none', 'pma'):
                self.pcl_projector = None
            else:
                raise ValueError(f'Unknown PCL projection topology {topology}')
        # Advance the reference's unused prediction decoder initialization before training and probes.
        Attention(dim)
        FeedForward(dim)

    def forward(self, x1: Tensor, x2: Tensor) -> SSLOutputs:
        """Return projected features used by HGCL, ICL and PCL."""
        one, two = self.encoder.two_views(x1, x2)
        tokens1, tokens2 = self.token_head(one.tokens), self.token_head(two.tokens)
        tokens1 = tokens1 * one.mask[:, 1:].unsqueeze(-1).to(tokens1.dtype)
        tokens2 = tokens2 * two.mask[:, 1:].unsqueeze(-1).to(tokens2.dtype)
        cls1, cls2 = self.cls_head(one.cls), self.cls_head(two.cls)
        pma1, pma2 = self.pma_head(one.pma_final.mean(1)), self.pma_head(two.pma_final.mean(1))
        return SSLOutputs(
            cls1,
            cls2,
            tokens1,
            tokens2,
            pma1,
            pma2,
            one.pma_states,
            two.pma_states,
            one.mask,
            two.mask,
            self.pma_head if self._pcl_uses_pma else self.pcl_projector,
        )


def build_encoder(cfg: Any) -> Encoder:
    """Build an encoder from resolved model and PMA configurations."""
    return Encoder(cfg)


class Pretrained(nn.Module):
    """Normalize raw [B, C, T] series with the stored TRAIN statistics, then encode."""

    def __init__(self, encoder: Encoder, mean: Tensor | None, std: Tensor | None) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer('mean', mean, persistent=False)
        self.register_buffer('std', std, persistent=False)

    def forward(self, x: Tensor) -> Features:
        if self.mean is not None:
            x = ((x - self.mean[:, None]) / self.std[:, None]).clamp(-10, 10)
        return self.encoder(x)


def parameter_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    """Return base, optimizer-gate and clipping-gate lists with the reference membership policies."""
    optimizer_names, clip_gates = set(), []
    for name, module in model.named_modules():
        is_gate = isinstance(module, (StateGate, SkipGate))
        if is_gate:
            clip_gates.extend(module.parameters())
        if is_gate or name.endswith(('state_update_gate', 'state_read_gate')):
            optimizer_names.update(f'{name}.{key}' if name else key for key, _ in module.named_parameters())
    base, optimizer_gates = [], []
    for name, parameter in model.named_parameters():
        (optimizer_gates if name in optimizer_names else base).append(parameter)
    return base, optimizer_gates, clip_gates
