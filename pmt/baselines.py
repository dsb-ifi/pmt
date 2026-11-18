"""Transformer-XL and vanilla xLSTM backbones in the PMA slot."""

# TXL arithmetic follows Hugging Face Transformers / Transformer-XL.
# Copyright 2018 Google AI, Google Brain, Carnegie Mellon University and NVIDIA.
# Licensed under Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_xlstm_started = False


class RelativeAttention(nn.Module):
    """Transformer-XL relative attention with a fresh zero memory each call."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim, self.head_dim = dim, dim // 8
        self.qkv_net = nn.Linear(dim, 3 * dim, bias=False)
        self.o_net = nn.Linear(dim, dim, bias=False)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-5)
        self.r_r_bias, self.r_w_bias = nn.Parameter(torch.zeros(8, dim // 8)), nn.Parameter(torch.zeros(8, dim // 8))
        self.r_net = nn.Linear(dim, dim, bias=False)
        self.drop, self.dropatt = nn.Dropout(0.1), nn.Dropout(0.1)

    def forward(self, x: Tensor, relative: Tensor, mask: Tensor, memory: int) -> Tensor:
        """Attend to causal input tokens and their initial zero-memory prefix."""
        length, batch, dim = x.shape
        joined = torch.cat((x.new_zeros(memory, batch, dim), x), 0) if memory else x
        query, key, value = self.qkv_net(joined).chunk(3, -1)
        query = query[-length:].view(length, batch, 8, self.head_dim)
        key, value = key.view(-1, batch, 8, self.head_dim), value.view(-1, batch, 8, self.head_dim)
        relative_key = self.r_net(relative).view(relative.size(0), 8, self.head_dim)
        content = torch.einsum('ibnd,jbnd->ijbn', (query + self.r_w_bias, key))
        positional = torch.einsum('ibnd,jnd->ijbn', (query + self.r_r_bias, relative_key))
        padded = torch.cat((positional.new_zeros(length, 1, batch, 8), positional), 1)
        positional = padded.view(positional.size(1) + 1, length, batch, 8)[1:].view_as(positional)
        scores = content + positional
        scores.mul_(1 / self.head_dim**0.5)
        scores = scores.float().masked_fill(mask[:, :, :, None], torch.finfo(scores.dtype).min).type_as(scores)
        probabilities = self.dropatt(F.softmax(scores, dim=1))
        attended = torch.einsum('ijbn,jbnd->ibnd', (probabilities, value)).contiguous().view(length, batch, dim)
        return self.layer_norm(x + self.drop(self.o_net(attended)))


class TXLFeedForward(nn.Module):
    """Post-normalized ReLU feedforward used by the paper's TXL comparator."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.CoreNet = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(4 * dim, dim), nn.Dropout(0.1)
        )
        self.layer_norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the original TXL residual and post normalization."""
        return self.layer_norm(x + self.CoreNet(x))


class TXLLayer(nn.Module):
    """One relative-attention Transformer-XL decoder layer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dec_attn, self.pos_ff = RelativeAttention(dim), TXLFeedForward(dim)

    def forward(self, x: Tensor, positions: Tensor, mask: Tensor, memory: int) -> Tensor:
        """Apply relative attention followed by the position-wise feedforward."""
        return self.pos_ff(self.dec_attn(x, positions, mask, memory))


class RelativePosition(nn.Module):
    """Transformer-XL concatenated sine/cosine relative positions."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer('inv_freq', 1 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim)))

    def forward(self, positions: Tensor) -> Tensor:
        """Return sequence-major relative position embeddings."""
        angles = torch.outer(positions, self.inv_freq)
        return torch.cat((angles.sin(), angles.cos()), -1)[:, None, :]


class TXLCore(nn.Module):
    """The inputs-embeds-only TXL path with no language-model dependency."""

    def __init__(self, dim: int, depth: int, memory: int) -> None:
        super().__init__()
        # Advance both initializers of the reference's unused one-row word embedding.
        word_embedding = torch.empty(1, dim).normal_()
        self.layers = nn.ModuleList([TXLLayer(dim) for _ in range(depth)])
        self.pos_emb, self.drop, self.memory = RelativePosition(dim), nn.Dropout(0.1), memory
        word_embedding.normal_(std=0.02)

        def initialize(module: nn.Module) -> None:
            # TXL initializes children before the attention module's two relative biases.
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                nn.init.normal_(module.weight, 1.0 if isinstance(module, nn.LayerNorm) else 0.0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, RelativeAttention):
                nn.init.normal_(module.r_w_bias, 0.0, 0.02)
                nn.init.normal_(module.r_r_bias, 0.0, 0.02)

        self.apply(initialize)

    def forward(self, x: Tensor) -> Tensor:
        """Reset memory each call, as in the released adapter."""
        x = x.transpose(0, 1).contiguous()
        length = x.size(0)
        positions = torch.arange(length + self.memory - 1, -1, -1.0, device=x.device, dtype=torch.int64).to(
            dtype=x.dtype
        )
        if self.memory > 0:
            positions.clamp_(max=self.memory)
        relative = self.pos_emb(positions)
        x, relative = self.drop(x), self.drop(relative)
        mask = torch.triu(x.new_ones((length, length + self.memory), dtype=torch.bool), diagonal=1 + self.memory)[
            :, :, None
        ]
        for layer in self.layers:
            x = layer(x, relative, mask, self.memory)
        return self.drop(x).transpose(0, 1).contiguous()


class Baseline(nn.Module):
    """Adapt recurrent-comparator token streams and EOS states to the encoder."""

    def __init__(self, cfg: Any) -> None:
        global _xlstm_started
        super().__init__()
        self.cfg = cfg.pma
        self.kind = cfg.pma.kind
        if self.kind == 'txl':
            self.model = TXLCore(cfg.model.dim, cfg.pma.layers, cfg.pma.window)
        elif self.kind == 'xlstm':
            from xlstm import (
                FeedForwardConfig,
                sLSTMBlockConfig,
                sLSTMLayerConfig,
                xLSTMBlockStack,
                xLSTMBlockStackConfig,
            )

            layer = sLSTMLayerConfig(
                backend='vanilla', num_heads=4, conv1d_kernel_size=4, bias_init='powerlaw_blockdependent', dropout=0.0
            )
            block = sLSTMBlockConfig(slstm=layer, feedforward=FeedForwardConfig(proj_factor=1.3, act_fn='gelu'))
            stack_cfg = xLSTMBlockStackConfig(
                embedding_dim=cfg.model.dim,
                context_length=cfg.pma.window,
                num_blocks=cfg.pma.layers,
                slstm_block=block,
                mlstm_block=None,
            )
            if not _xlstm_started:
                from xlstm.blocks.slstm.layer import sLSTMLayer

                # Replay the first layer's draws before the reference's failed CUDA backend load.
                block._block_idx = 0
                block.__post_init__()
                sLSTMLayer(block.slstm)
                _xlstm_started = True
            self.stack = xLSTMBlockStack(stack_cfg)
        else:
            raise ValueError(f'Unsupported baseline {self.kind}')

    def forward(self, x: Tensor, mask: Tensor, *, return_all_states: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """Return encoded tokens, EOS state and the adapter's reference state history."""
        x = self.model(x) if self.kind == 'txl' else self.stack(x)
        final = x[:, -1:, :]
        # TXL always returns a history. xLSTM also does when cue evaluation requests it.
        states = (
            x.unfold(1, self.cfg.window, self.cfg.stride)[..., -1].unsqueeze(2)
            if self.kind == 'txl' or (return_all_states and x.size(1) >= self.cfg.window)
            else final.unsqueeze(2)
        )
        return x, final, states


def build_baseline(cfg: Any) -> Baseline:
    """Build the selected paper comparator without importing optional xLSTM early."""
    return Baseline(cfg)
