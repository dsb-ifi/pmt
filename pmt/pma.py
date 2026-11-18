"""Progressive memory attention with overlapping or shifted disjoint windows."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def norm(dim: int) -> nn.Module:
    """Return the paper's RMS normalization."""
    return nn.RMSNorm(dim, eps=1e-5)


def sanitize(mask: Tensor) -> Tensor:
    """Apply the reference self-attention mask's diagonal and two-key openings."""
    mask = mask.clone()
    diagonal = torch.arange(min(mask.shape[-2:]), device=mask.device)
    mask[..., diagonal, diagonal] = False
    blocked = mask.all(-1)
    mask[..., :, -1] = mask[..., :, -1] & ~blocked
    one = (~mask).sum(-1) == 1
    mask[..., :, 0] = mask[..., :, 0] & ~one
    one = (~mask).sum(-1) == 1
    if mask.size(-1) > 1:
        mask[..., :, 1] = mask[..., :, 1] & ~one
    return mask


class Attention(nn.Module):
    """Fused QKV attention with optional per-head, state-excluding rotary positions."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask: Tensor, positions: Tensor | None = None) -> Tensor:
        """Convert the sanitized block mask and attend with optional absolute token positions."""
        batch, length, dim = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, -1)
        if positions is not None:
            half = dim // 16
            inv = 10000.0 ** (-torch.arange(half, dtype=torch.float32, device=x.device) / half)
            angles = positions.clamp_min(0).float().unsqueeze(-1) * inv.unsqueeze(0)
            sin, cos = angles.sin().to(q.dtype).unsqueeze(1), angles.cos().to(q.dtype).unsqueeze(1)
            valid = (positions >= 0).view(length, 1, 1)
            rotated = []
            for item in (q, k):
                item = item.reshape(batch, length, 8, dim // 8).permute(1, 0, 2, 3).reshape(length, batch * 8, dim // 8)
                left, right = item[..., :half], item[..., half:]
                item = torch.cat(
                    (
                        torch.where(valid, left * cos - right * sin, left),
                        torch.where(valid, left * sin + right * cos, right),
                    ),
                    -1,
                )
                rotated.append(
                    item.reshape(length, batch, 8, dim // 8)
                    .permute(1, 0, 2, 3)
                    .reshape(batch, length, dim)
                    .contiguous()
                )
            q, k = rotated
        q, k, v = (item.reshape(batch, length, 8, dim // 8).transpose(1, 2) for item in (q, k, v))
        negative = -6e4 if q.dtype in (torch.float16, torch.bfloat16) else -1e9
        additive = torch.where(
            mask, torch.full((), negative, dtype=q.dtype, device=mask.device), torch.zeros_like(mask, dtype=q.dtype)
        )
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=additive, dropout_p=0.1 if self.training else 0.0)
        return self.o_proj(result.transpose(1, 2).contiguous().reshape(batch, length, dim))


class WindowLayer(nn.Module):
    """Pre-normalized window transformer preserving sequence-major dropout order."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_attn = Attention(dim)
        self.linear1, self.linear2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)
        self.norm1, self.norm2 = norm(dim), norm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor, mask: Tensor, positions: Tensor) -> Tensor:
        """Transform a batched window with its two state prefixes."""
        src = x.transpose(0, 1)
        attention = self.self_attn(self.norm1(src).transpose(0, 1), mask, positions)
        src = src + self.dropout(attention.transpose(0, 1))
        update = self.linear1(self.norm2(src))
        update = self.dropout(F.gelu(update))
        update = self.dropout(self.linear2(update))
        return (src + update).transpose(0, 1)


class _ZeroInitLinear(nn.Linear):
    """Initialize a tokenwise gate without advancing the random stream."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


class StateGate(nn.Module):
    """Mix fresh and previous-block states with the released checkpoint topology."""

    def __init__(self, dim: int, cfg: Any) -> None:
        super().__init__()
        self.kind, self.residual = cfg.gate, cfg.gate_residual
        if self.kind == 'scalar':
            self.scalar_param = nn.Parameter(torch.full((1,), float(cfg.skip_init)))
        elif self.kind == 'tokenwise':
            self.tokenwise_linear = _ZeroInitLinear(dim, dim)
            nn.init.constant_(self.tokenwise_linear.bias, cfg.skip_init)
        else:
            raise ValueError(f'Unsupported state gate {self.kind}')

    def forward(self, previous: Tensor, fresh: Tensor) -> Tensor:
        """Preserve the reference interpolation and infinity handling."""
        gate = torch.sigmoid(self.scalar_param if self.kind == 'scalar' else self.tokenwise_linear(previous))
        if self.residual:
            previous = torch.where(torch.isinf(previous), torch.zeros_like(previous), previous)
            fresh = torch.where(torch.isinf(fresh), torch.zeros_like(fresh), fresh)
            return previous + gate * (fresh - previous)
        return gate * previous + (1.0 - gate) * fresh


class SkipGate(nn.Module):
    """Add a learned gated update to a normalized residual stream."""

    def __init__(self, dim: int, cfg: Any) -> None:
        super().__init__()
        self.kind = cfg.skip_gate
        if self.kind == 'scalar':
            self.param = nn.Parameter(torch.full((1,), float(cfg.skip_init)))
        elif self.kind == 'tokenwise':
            self.param = _ZeroInitLinear(dim, dim)
            nn.init.constant_(self.param.bias, cfg.skip_init)
        elif self.kind == 'tokenwise_legacy_constant_weight':
            self.param = nn.Linear(dim, dim, bias=False)
            nn.init.constant_(self.param.weight, cfg.skip_init / math.sqrt(dim))
        else:
            raise ValueError(f'Unsupported skip gate {self.kind}')

    def forward(self, skip: Tensor, update: Tensor) -> Tensor:
        """Return the paper's additive skip plus sigmoid-gated update."""
        logit = self.param.view(1, 1, -1) if self.kind == 'scalar' else self.param(skip)
        return skip + torch.sigmoid(logit) * update


class Overlap(nn.Module):
    """Pool valid overlap slots using a learned query and relative-position bias."""

    def __init__(self, dim: int, cfg: Any) -> None:
        super().__init__()
        self.ln, self.post_agg_ln = nn.LayerNorm(dim), nn.LayerNorm(dim)
        if cfg.conv_query:
            self.q_conv = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        hidden = max(dim // 4, 64)
        self.q_mlp = nn.Sequential(nn.Linear(dim, hidden, bias=False), nn.GELU(), nn.Linear(hidden, dim, bias=False))
        self.k_proj, self.v_proj = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)
        self.rel_bias = nn.Parameter(torch.zeros(8, math.ceil(cfg.window / cfg.stride)))
        self.gamma = nn.Parameter(torch.full((1,), float(cfg.gamma)))

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        """Merge `[B,L,O,D]` slots with the original attention arithmetic."""
        batch, length, slots, dim = x.shape
        x = x.masked_fill(~valid[..., None, None], 0.0)
        present = x.abs().sum(-1) > 0
        normalized = self.ln(x)
        query_input = normalized
        if hasattr(self, 'q_conv'):
            query_input = query_input.permute(0, 1, 3, 2).reshape(batch * length, dim, slots)
            query_input = self.q_conv(query_input).reshape(batch, length, dim, slots).permute(0, 1, 3, 2)
        weights = present.to(normalized.dtype).unsqueeze(-1)
        denominator = present.to(normalized.dtype).sum(2, keepdim=True).clamp_min(1.0).unsqueeze(-1)
        query = self.q_mlp((query_input * weights).sum(2) / denominator.squeeze(-1)).view(
            batch * length, 8, 1, dim // 8
        )
        key, value = (
            projection(normalized)
            .reshape(batch, length, slots, 8, dim // 8)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch * length, 8, slots, dim // 8)
            for projection in (self.k_proj, self.v_proj)
        )
        keep = present.view(batch * length, 1, slots, 1).clone()
        empty = keep.squeeze(-1).squeeze(1).sum(-1) == 0
        keep[:, 0, 0, 0] = keep[:, 0, 0, 0] | empty
        key, value = key.masked_fill(~keep, 0.0), value.masked_fill(~keep, 0.0)
        token_keep = valid.unsqueeze(-1).expand(-1, -1, 8).reshape(batch * length, 8, 1, 1)
        key, value = key.masked_fill(~token_keep, 0.0), value.masked_fill(~token_keep, 0.0)
        additive = torch.zeros_like(keep.squeeze(-1).unsqueeze(2), dtype=normalized.dtype)
        additive = additive.masked_fill(~keep.squeeze(-1).unsqueeze(2), float('-inf'))
        bias = self.rel_bias[:, :slots].unsqueeze(0).unsqueeze(2).to(normalized.dtype)
        if query.size(0) > 65535:
            result = torch.cat(
                [
                    F.scaled_dot_product_attention(q, k, v, attn_mask=bias + mask, dropout_p=0.0)
                    for q, k, v, mask in zip(
                        query.split(65535), key.split(65535), value.split(65535), additive.split(65535)
                    )
                ],
                dim=0,
            )
        else:
            result = F.scaled_dot_product_attention(query, key, value, attn_mask=bias + additive, dropout_p=0.0)
        result = result.squeeze(2).reshape(batch, length, dim).masked_fill(~valid.unsqueeze(-1), 0.0)
        counts = present.sum(2).clamp_min(1.0).to(x.dtype)
        return self.gamma * self.post_agg_ln(result * counts.rsqrt().unsqueeze(-1))


class WindowProcessor(nn.Module):
    """Carry horizontal states and route previous-block states through local windows."""

    def __init__(self, dim: int, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.shifted = cfg.kind == 'spma'
        self.sos_state_tokens = nn.Parameter(torch.randn(cfg.states, dim) * 0.02)
        self.base_state = nn.Parameter(torch.randn(cfg.states, dim) * 0.02)
        self.state_norm, self.gate = norm(dim), StateGate(dim, cfg)
        if self.shifted:
            self.state_update_gate, self.state_read_gate = nn.Linear(3 * dim, 1), nn.Linear(dim, 1)
            self.state_read_proj = nn.Linear(dim, dim)
            for gate in (self.state_update_gate, self.state_read_gate):
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, -2.0)
            nn.init.zeros_(self.state_read_proj.weight)
            nn.init.zeros_(self.state_read_proj.bias)
        self.layer = WindowLayer(dim)

    def _route(self, previous: Tensor, old: int, new: int, count: int) -> Tensor:
        """Blend old-offset states into new windows, with SOS on the left and zeros on the right."""
        batch = previous.size(0)
        left = self.sos_state_tokens.to(previous).view(1, 1, *self.sos_state_tokens.shape).expand(batch, 1, -1, -1)
        bank = torch.cat((left, previous, torch.zeros_like(left)), 1)
        index = torch.arange(count, device=previous.device)
        delta = new - old
        alpha = abs(delta) / self.cfg.window
        if delta > 0:
            return alpha * bank.index_select(1, index.clamp(0, bank.size(1) - 1)) + (1.0 - alpha) * bank.index_select(
                1, (index + 1).clamp(0, bank.size(1) - 1)
            )
        return (1.0 - alpha) * bank.index_select(1, (index + 1).clamp(0, bank.size(1) - 1)) + alpha * bank.index_select(
            1, (index + 2).clamp(0, bank.size(1) - 1)
        )

    def forward(
        self, x: Tensor, previous: Tensor | None, offset: int, old_offset: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return window outputs and both final and per-window state tensors."""
        batch, length, dim = x.shape
        cfg, device = self.cfg, x.device
        width, states = cfg.window, cfg.states
        stride = width if self.shifted else cfg.stride
        padded_length = (
            math.ceil((length + offset) / width) * width
            if self.shifted
            else math.ceil((length - width) / stride) * stride + width
        )
        front = offset if self.shifted else (padded_length - length) // 2
        if padded_length < width:
            raise ValueError('Sequence cannot form a memory window')
        padded = F.pad(x, (0, 0, front, padded_length - length - front))
        windows = padded.unfold(1, width, stride).permute(0, 1, 3, 2).contiguous()
        count = windows.size(1)
        if self.shifted and previous is not None:
            previous = self._route(previous, old_offset, offset, count)
        carry = self.sos_state_tokens.unsqueeze(0).expand(batch, -1, -1).clone()
        base = self.base_state.unsqueeze(0).expand(batch, -1, -1)
        if self.shifted:
            base = base.contiguous()
        noise = None
        if self.training and cfg.state_noise > 0 and not self.shifted:
            noise = cfg.state_noise * torch.randn(batch, count, states, dim, device=device, dtype=base.dtype)
        base_mask = torch.zeros(2 * states + width, 2 * states + width, dtype=torch.bool, device=device)
        base_mask[2 * states :, 2 * states :] = torch.triu(torch.ones(width, width, dtype=torch.bool, device=device), 1)
        base_mask[2 * states :, states : 2 * states] = True
        outputs, all_states = [], []
        if self.shifted:
            windows_out = windows.new_empty(batch, count, width, dim)
            history = carry.new_empty(batch, count, states, dim)
        for index in range(count):
            fresh = base
            if noise is not None:
                fresh = fresh + noise[:, index]
            if self.shifted and self.training and cfg.state_noise > 0:
                fresh = fresh + cfg.state_noise * torch.randn(batch, states, dim, device=device, dtype=base.dtype)
            current = self.state_norm(fresh if previous is None else self.gate(previous[:, index], fresh))
            positions = index * stride + torch.arange(width, device=device)
            valid = (positions >= front) & (positions < front + length)
            has_padding = not (index * stride >= front and index * stride + width <= front + length)
            rope_positions = positions
            if self.shifted:
                rope_positions = (positions - front).masked_fill(~valid, -1)
            rope_positions = torch.cat((torch.full((2 * states,), -1, device=device), rope_positions))
            blocked = base_mask
            if has_padding:
                padded_rows = torch.cat((torch.zeros(2 * states, dtype=torch.bool, device=device), ~valid))
                blocked = blocked | padded_rows.unsqueeze(1) | padded_rows.unsqueeze(0)
            blocked = sanitize(blocked)
            token_window = windows[:, index]
            attention_carry, attention_current = carry, current
            if (
                self.shifted
                and torch.is_autocast_enabled(device.type)
                and token_window.dtype in (torch.float16, torch.bfloat16)
            ):
                attention_carry, attention_current = carry.to(token_window.dtype), current.to(token_window.dtype)
            transformed = self.layer(
                torch.cat((attention_carry, attention_current, token_window), 1), blocked, rope_positions
            )
            if has_padding:
                mask_dtype = transformed.dtype if self.shifted else x.dtype
                transformed = transformed * torch.cat(
                    (torch.ones(2 * states, device=device, dtype=mask_dtype), valid.to(mask_dtype))
                ).view(1, -1, 1)
            candidate = transformed[:, states : 2 * states]
            if self.shifted:
                candidate = candidate.to(current.dtype)
            candidate = self.state_norm(candidate)
            raw = transformed[:, 2 * states :]
            output = raw
            if self.shifted:
                normalized_carry = self.state_norm(carry)
                read_source = normalized_carry.mean(1)
                read_gate = torch.sigmoid(self.state_read_gate(read_source)).view(batch, 1, 1)
                output = raw + read_gate * self.state_read_proj(read_source).unsqueeze(1)
                if has_padding:
                    output = output * valid.to(output.dtype).view(1, -1, 1)
                    token_mask = valid.to(raw.dtype).view(1, -1, 1)
                    token_mean = (raw * token_mask).sum(1) / token_mask.sum(1).clamp_min(1.0)
                else:
                    token_mean = raw.mean(1)
                # The read and update paths normalize independently in the reference graph.
                normalized_carry = self.state_norm(carry)
                gate_input = torch.cat((normalized_carry.mean(1), candidate.mean(1), token_mean), -1)
                update = torch.sigmoid(self.state_update_gate(gate_input) / 2.0).view(batch, 1, 1)
                carry = self.state_norm(normalized_carry + update * (candidate - normalized_carry))
            else:
                carry = candidate
            if self.shifted:
                windows_out[:, index], history[:, index] = output, carry
            else:
                outputs.append(output)
                all_states.append(carry.unsqueeze(1))
        if not self.shifted:
            windows_out = torch.stack(outputs, 1)
            history = torch.cat(all_states, 1)
        final = history[:, -1] if self.shifted else F.layer_norm(carry.clone(), carry.shape[-1:])
        if self.shifted:
            return (
                windows_out.reshape(batch, padded_length, dim)[:, front : front + length].contiguous(),
                final,
                history,
            )
        positions = torch.arange(count, device=device)[:, None] * stride + torch.arange(width, device=device)[None, :]
        # Rank occurrences of each token in sorted window order to assign its overlap slots.
        values, order = positions.flatten().sort()
        start = torch.ones_like(values, dtype=torch.bool)
        start[1:] = values[1:] != values[:-1]
        sorted_position = torch.arange(values.numel(), device=device)
        first = torch.where(start, sorted_position, torch.zeros_like(sorted_position)).cummax(0).values
        ranks = torch.empty_like(order)
        ranks[order] = sorted_position - first
        overlap = max(1, min(count, math.ceil(width / stride)))
        assembled = x.new_zeros(batch, padded_length, overlap, dim)
        batch_index = torch.arange(batch, device=device)[:, None, None].expand(-1, count, width)
        token_index = positions.unsqueeze(0).expand(batch, -1, -1)
        slot_index = ranks.view(count, width).unsqueeze(0).expand(batch, -1, -1)
        assembled.index_put_(
            (batch_index.flatten(), token_index.flatten(), slot_index.flatten()),
            windows_out.reshape(-1, dim).to(assembled.dtype),
            accumulate=False,
        )
        return assembled[:, front : front + length].contiguous(), final, history


class MemoryBlock(nn.Module):
    """One memory-window layer followed by overlap fusion and an outer MLP."""

    def __init__(self, dim: int, cfg: Any) -> None:
        super().__init__()
        self.shifted = cfg.kind == 'spma'
        self.norm1, self.norm2, self.norm3 = norm(dim), norm(dim), norm(dim)
        self.chunk_processor = WindowProcessor(dim, cfg)
        if not self.shifted:
            self.aggregator = Overlap(dim, cfg)
        self.skip_gate, self.resid_ln = SkipGate(dim, cfg), norm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(4 * dim, dim), nn.Dropout(0.1)
        )

    def forward(
        self, x: Tensor, mask: Tensor, previous: Tensor | None, offset: int, old: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Fuse local attention with the original PMA or sPMA residual semantics."""
        normalized = self.norm1(x)
        output, final, history = self.chunk_processor(normalized, previous, offset, old)
        if self.shifted:
            # Preserve reference node order for exact accumulation at norm1's output.
            update = output - normalized
            skip = self.resid_ln(normalized)
            fused = self.skip_gate(skip, update) - skip
        else:
            present = (output.abs().sum(-1) > 0) & mask.unsqueeze(2)
            weights = present.to(output.dtype).unsqueeze(-1)
            skip = (output * weights).sum(2) / weights.sum(2).clamp_min(1.0)
            fused = self.skip_gate(self.resid_ln(skip), self.aggregator(output, mask))
        output = self.norm2(x + fused)
        output = output + self.mlp(output)
        return self.norm3(output), final, history


class PMA(nn.Module):
    """Stack memory blocks with horizontal recurrence and cross-block window states."""

    def __init__(self, dim: int, cfg: Any) -> None:
        super().__init__()
        if cfg.kind not in ('pma', 'spma') or cfg.layers < 1 or cfg.states < 1:
            raise ValueError('PMA requires a supported kind and positive depth and state count')
        if cfg.window < 1 or cfg.stride < 1 or cfg.stride > cfg.window:
            raise ValueError('PMA requires 1 <= stride <= window')
        self.cfg = cfg
        self.blocks = nn.ModuleList([MemoryBlock(dim, cfg) for _ in range(cfg.layers)])

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return encoded tokens, final states and final-block window-state history."""
        previous, old = None, 0
        for index, block in enumerate(self.blocks):
            offset = self.cfg.window // 2 if self.cfg.kind == 'spma' and index % 2 else 0
            x, final, previous = block(x, mask, previous, offset, old)
            old = offset
        return x, final, previous
