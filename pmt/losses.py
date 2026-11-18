"""Publication HGCL, PCL and ICL with reproducible negative sampling."""

from __future__ import annotations

import random
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _logits(similarity: Tensor, tau: float | Tensor) -> Tensor:
    return (similarity.float() / tau).clamp(-100.0, 100.0)


def _windows(x: Tensor, width: int, stride: int, tail: bool) -> Tensor:
    offset = (x.size(1) - width) % stride if tail else 0
    return x[:, offset:].unfold(1, width, stride).permute(0, 1, 3, 2).contiguous()


def _bucket(positive: Tensor, aggregate: Tensor, negatives: Tensor, tau: float) -> Tensor:
    # Strict alignment is the numerator; the weighted neighborhood is one extra logit,
    # not a weighted multi-positive numerator. Clamp each score before subtracting.
    pos = _logits(positive, tau)
    soft = _logits(aggregate, tau) - pos
    logits = torch.cat([pos.unsqueeze(1), soft.unsqueeze(1), _logits(negatives, tau)], dim=1)
    return (torch.logsumexp(logits, dim=1) - pos).mean()


def _patch_indices(
    rows: Tensor,
    count: int,
    width: int,
    requested: int,
    token_radius: int,
    window_radius: int,
    generator: torch.Generator,
) -> Tensor:
    """Sample the oracle's excluded-interval bank with its two geometry-dependent RNG paths."""
    per_sample = count * width
    aw, ap = (rows % per_sample) // width, rows % width
    if rows.numel() * per_sample <= 8_000_000:
        local = torch.arange(per_sample, device=rows.device).unsqueeze(0)
        windows, tokens = local // width, local % width
        delta = (windows - aw.unsqueeze(1)).abs()
        exclude = (delta == 0) & ((tokens - ap.unsqueeze(1)).abs() <= token_radius)
        exclude = exclude | ((delta <= window_radius) & (delta > 0))
        available = ~exclude
        k = min(requested, int(available.sum(dim=1).min()))
        selected = torch.multinomial(available.float(), num_samples=k, replacement=False, generator=generator)
    else:
        left_lo = (aw - window_radius).clamp(min=0) * width
        left_len = aw * width - left_lo
        anchor_lo = aw * width + (ap - token_radius).clamp(min=0, max=width - 1)
        anchor_hi = aw * width + (ap + token_radius).clamp(min=0, max=width - 1) + 1
        anchor_len = (anchor_hi - anchor_lo).clamp(min=1)
        right_lo = (aw + 1).clamp(max=count) * width
        right_len = (aw + window_radius + 1).clamp(max=count) * width - right_lo
        available = per_sample - left_len - anchor_len - right_len
        k = min(requested, int(available.min()))
        draws = torch.empty((rows.numel(), k), dtype=torch.long, device=rows.device)
        block = max(128, min(rows.numel(), (64 * 1024 * 1024) // (max(int(available.max()), 1) * 4)))
        for begin in range(0, rows.numel(), block):
            avail = available[begin : begin + block]
            keys = torch.rand((avail.numel(), int(avail.max())), device=rows.device, generator=generator)
            if int(avail.min()) != int(avail.max()):
                keys = keys.masked_fill(
                    torch.arange(keys.size(1), device=rows.device).unsqueeze(0) >= avail.unsqueeze(1), -1.0
                )
            draws[begin : begin + block] = torch.topk(keys, k=k, dim=1, largest=True, sorted=False).indices
        selected = (
            draws
            + (draws >= left_lo.unsqueeze(1)).long() * left_len.unsqueeze(1)
            + (draws >= (anchor_lo - left_len).unsqueeze(1)).long() * anchor_len.unsqueeze(1)
            + (draws >= (right_lo - left_len - anchor_len).unsqueeze(1)).long() * right_len.unsqueeze(1)
        )
    return (rows // per_sample).unsqueeze(1) * per_sample + selected


def _indexed_dot(anchors: Tensor, bank: Tensor, indices: Tensor) -> Tensor:
    # The fixed 64-column blocking is part of the reference's floating-point reduction path.
    return torch.cat(
        [
            torch.bmm(bank[indices[:, begin : begin + 64]], anchors.unsqueeze(-1)).squeeze(-1)
            for begin in range(0, indices.size(1), 64)
        ],
        dim=1,
    )


def _hgcl(
    z1: Tensor, z2: Tensor, mask1: Tensor | None, mask2: Tensor | None, cfg: Any, step: int
) -> tuple[Tensor, Tensor]:
    """Return scalar bucket HGCL and RNG state for [B,T,D] views with step-seeded negatives."""
    if z1.shape != z2.shape or z1.ndim != 3:
        raise ValueError('HGCL requires matching [batch, tokens, features] tensors')
    width, stride = cfg.window, cfg.stride
    if width < 1 or stride < 1 or z1.size(1) < width or cfg.negatives < 1:
        raise ValueError('Invalid HGCL window geometry or negative count')
    # CPU normalization keeps the input dtype; leave bank matmuls autocast-eligible
    # and promote scores only afterwards in _logits, exactly as the reference does.
    projected1 = z1
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    zero = projected1.sum() * 0
    # Keep the reference's full-batch chunk nodes: backward accumulates in input dtype.
    z1, z2 = z1[:].contiguous(), z2[:].contiguous()
    a, b = _windows(z1, width, stride, cfg.tail), _windows(z2, width, stride, cfg.tail)
    if mask1 is None and mask2 is None:
        mask1 = torch.ones(z1.shape[:2], dtype=torch.bool, device=z1.device)
        mask2 = mask1
    if mask1 is None or mask2 is None:
        raise ValueError('HGCL needs both view masks or neither')
    m1 = _windows(mask1.unsqueeze(-1).float(), width, stride, cfg.tail).squeeze(-1)
    m2 = _windows(mask2.unsqueeze(-1).float(), width, stride, cfg.tail).squeeze(-1)
    batch, count, _, dim = a.shape
    # Built on the CPU and moved, as the reference's registered buffer: CUDA exp can differ in the last bit.
    indices = torch.arange(width, dtype=torch.float32)
    distance = (indices.unsqueeze(0) - indices.unsqueeze(1)).abs()
    if cfg.kernel == 'gaussian':
        kernel = torch.exp(-(distance**2) / (2 * ((width / 5.0) * cfg.width_local) ** 2))
    elif cfg.kernel == 'triangular':
        kernel = (1.0 - distance / float(cfg.radius_patch + 1)).clamp_min(0.0)
    else:
        raise ValueError(f'Unknown HGCL kernel: {cfg.kernel}')
    kernel = kernel.to(z1.device)
    # Patch keys stay attached. Create this fork before local similarities so
    # their BF16 backward contributions accumulate in the reference order.
    bank = b.view(-1, dim)
    anchor_index = torch.arange(0, count, getattr(cfg, 'anchor_stride', 1), device=a.device)
    aa, ab, am1, am2 = a, b, m1, m2
    if getattr(cfg, 'anchor_stride', 1) > 1 and count > 1:
        aa, ab, am1, am2 = a[:, anchor_index], b[:, anchor_index], m1[:, anchor_index], m2[:, anchor_index]
    local = torch.matmul(aa, ab.transpose(-1, -2)).reshape(-1, width)
    flat_anchors = aa.view(-1, dim)
    rows = torch.nonzero(((am1 > 0.5) & (am2 > 0.5)).reshape(-1), as_tuple=True)[0]
    if not rows.numel():
        raise ValueError('HGCL has no anchors valid in both views')
    positions = rows % width
    selected_local = local[rows]
    q = kernel.index_select(0, positions)
    candidate_valid = (am2 > 0.5).reshape(-1, width).index_select(0, rows // width)
    q = q * candidate_valid.to(q.dtype)
    keep = q.sum(dim=-1) > 0
    rows, positions, q = rows[keep], positions[keep], q[keep]
    anchors = flat_anchors[rows]
    local = selected_local[keep]
    positive = local.gather(1, positions.unsqueeze(1)).squeeze(1)
    sample_ids = rows // (aa.size(1) * width)
    full_rows = sample_ids * count * width + anchor_index[(rows % (aa.size(1) * width)) // width] * width + positions
    generator = torch.Generator(device=z1.device).manual_seed(step)
    n_in = int(round(cfg.patch_in_fraction * cfg.negatives))
    n_out = cfg.negatives - n_in or cfg.negatives
    bank_indices = torch.randint(0, bank.size(0), (min(n_out, bank.size(0)),), device=z1.device, generator=generator)
    negatives = torch.matmul(anchors, bank[bank_indices].t())
    same_sample = sample_ids.unsqueeze(1) == (bank_indices // (count * width)).unsqueeze(0)
    bank_valid = (m2 > 0.5).reshape(-1)[bank_indices]
    negatives = negatives.masked_fill(same_sample | ~bank_valid.unsqueeze(0), float('-inf'))
    if n_in:
        # Same-sample draws restart too; these are not one advancing RNG stream.
        generator.manual_seed(step)
        token_radius = (
            cfg.exclude_patch
            if cfg.exclude_patch is not None
            else (cfg.radius_patch if cfg.radius_patch is not None else width)
        )
        window_radius = cfg.exclude_window if cfg.exclude_window is not None else (cfg.radius_window or 0)
        selected = _patch_indices(full_rows, count, width, n_in, token_radius, window_radius, generator)
        if selected.size(1):
            extra = _indexed_dot(anchors, bank, selected)
            extra = extra.masked_fill(~(m2 > 0.5).reshape(-1)[selected], float('-inf'))
            negatives = torch.cat([negatives, extra], dim=1)
    patch_loss = _bucket(positive, (local * q).sum(dim=1), negatives, cfg.patch_tau)
    # Preserve the reference valid-anchor weighted reduction for the single publication chunk.
    patch_count = float(rows.numel())
    patch_loss = (zero + patch_loss * patch_count) / patch_count
    if cfg.window_weight == 0:
        return patch_loss, generator.get_state()

    # Masked means are formed from normalized patches, then normalized again.
    wa = (a * m1.unsqueeze(-1)).sum(2) / m1.sum(2).clamp_min(1e-6).unsqueeze(-1)
    wb = (b * m2.unsqueeze(-1)).sum(2) / m2.sum(2).clamp_min(1e-6).unsqueeze(-1)
    wa, wb = F.normalize(wa, dim=-1), F.normalize(wb, dim=-1)
    # Keep the aligned-window branch before the neighborhood convolution, as in
    # the oracle: branch order matters where gradients meet a BF16 unfold.
    # Even the identity selections retain that reference backward graph.
    window_indices = torch.arange(count, device=wa.device)
    wa_anchor = wa.index_select(1, window_indices)
    wb_anchor = wb.index_select(1, window_indices)
    aligned = (wa_anchor * wb_anchor).sum(dim=-1).reshape(-1)
    valid1, valid2 = m1.any(-1), m2.any(-1)
    radius = min(cfg.radius_window, count - 1) if cfg.radius_window is not None else count - 1
    offsets = torch.arange(-radius, radius + 1, dtype=wa.dtype, device=wa.device).abs()
    if cfg.kernel == 'triangular':
        wk = (1.0 - offsets / float(cfg.radius_window + 1)).clamp_min(0.0)
    else:
        sigma = (width / 2.5) * (max(width - stride, 0) / width) * cfg.width_window
        wk = torch.exp(-((offsets * float(stride)) ** 2) / (2.0 * max(sigma**2, 1e-6)))
    q_mass = F.conv1d(valid2.to(wa.dtype).unsqueeze(1), wk.view(1, 1, -1), padding=radius).squeeze(1)
    wrows = torch.nonzero((valid1 & valid2 & (q_mass > 0)).reshape(-1), as_tuple=True)[0]
    weighted = wb * valid2.unsqueeze(-1).to(wb.dtype)
    filters = wk.view(1, 1, -1).expand(dim, 1, -1).contiguous()
    aggregate = F.conv1d(weighted.transpose(1, 2), filters, padding=radius, groups=dim).transpose(1, 2)
    positive = aligned[wrows]
    total = (wa_anchor * aggregate).sum(dim=-1).reshape(-1)[wrows]
    # Only window-bank keys stop gradient; local positives still train view 2.
    bank = wb.view(-1, dim).detach()
    # Patch and window banks deliberately restart from the same training step.
    generator.manual_seed(step)
    n_in = int(round(cfg.window_in_fraction * cfg.negatives))
    n_out = cfg.negatives - n_in or cfg.negatives
    bank_indices = torch.randint(0, bank.size(0), (min(n_out, bank.size(0)),), device=wa.device, generator=generator)
    negatives = torch.matmul(wa_anchor.view(-1, dim)[wrows], bank[bank_indices].t())
    same_sample = (wrows // count).unsqueeze(1) == (bank_indices // count).unsqueeze(0)
    negatives = negatives.masked_fill(same_sample, float('-inf'))
    if n_in:
        radius = cfg.exclude_window if cfg.exclude_window is not None else (cfg.radius_window or 0)
        left, right = (wrows % count - radius).clamp(min=0), (wrows % count + radius).clamp(max=count - 1)
        excluded = (right + 1 - left).clamp(min=1)
        available_min = int((count - excluded).clamp(min=0).min())
        n_in = min(n_in, available_min)
        generator.manual_seed(step)
        sampled = torch.randint(0, available_min, (wrows.numel(), n_in), device=wa.device, generator=generator)
        selected = (
            (wrows // count).unsqueeze(1) * count + sampled + (sampled >= left.unsqueeze(1)) * excluded.unsqueeze(1)
        )
        if n_in:
            negatives = torch.cat([negatives, _indexed_dot(wa_anchor.view(-1, dim)[wrows], bank, selected)], dim=1)
    window_loss = _bucket(positive, total, negatives, cfg.window_tau)
    window_count = float(wrows.numel())
    window_loss = (zero + window_loss * window_count) / window_count
    return patch_loss + window_loss * cfg.window_weight, generator.get_state()


class _SampledNCE(torch.autograd.Function):
    """Keep the reference compact backward statistics and stop-gradient keys."""

    @staticmethod
    def forward(ctx: Any, anchors: Tensor, positives: Tensor, negatives: Tensor, tau: Tensor) -> Tensor:
        a, p, n = anchors.float(), positives.float(), negatives.float()
        pos = (a * p).sum(dim=1)
        neg = torch.bmm(n, a.unsqueeze(-1)).squeeze(-1)
        pos_logits, neg_logits = _logits(pos.unsqueeze(1), tau).squeeze(1), _logits(neg, tau)
        log_z = torch.logaddexp(pos_logits, torch.logsumexp(neg_logits, dim=1))
        pos_mask = ((pos / tau >= -100.0) & (pos / tau <= 100.0)).to(a.dtype)
        neg_mask = ((neg / tau >= -100.0) & (neg / tau <= 100.0)).to(a.dtype)
        probs = torch.softmax(torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1), dim=1)
        expected = p * (pos_mask * probs[:, 0]).unsqueeze(1)
        expected = expected + torch.bmm((neg_mask * probs[:, 1:]).unsqueeze(1), n).squeeze(1)
        ctx.save_for_backward(expected, p * pos_mask.unsqueeze(1), tau)
        ctx.anchor_dtype, ctx.rows = anchors.dtype, anchors.size(0)
        return (log_z - pos_logits).mean()

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None, None, None]:
        expected, positive, tau = ctx.saved_tensors
        derivative = ((expected - positive) / tau) / float(ctx.rows)
        return (derivative * grad.to(derivative.dtype)).to(ctx.anchor_dtype), None, None, None


def _pcl_direction(a: Tensor, b: Tensor, cfg: Any, rows_per_sample: int, generator: torch.Generator) -> Tensor:
    rows, dim = a.shape
    indices = torch.arange(rows, dtype=torch.long, device=a.device)
    excluded = rows_per_sample if cfg.exclude_sample else 1
    if rows <= excluded:
        raise ValueError('PCL has no eligible negative states')
    choices = torch.randint(0, rows - excluded, (rows, cfg.negatives), device=a.device, generator=generator)
    # Pre-sample all rows before chunking: chunk size changes memory/reduction,
    # while negative identities and private RNG consumption stay unchanged.
    start = (indices // rows_per_sample) * rows_per_sample if cfg.exclude_sample else indices
    choices = choices + (choices >= start.unsqueeze(1)) * excluded
    pool = b.detach()
    tau = torch.tensor(float(cfg.tau), device=a.device)
    losses, weights = [], []
    for begin in range(0, rows, max(1, int(cfg.chunk))):
        end = min(begin + max(1, int(cfg.chunk)), rows)
        neg = pool.index_select(0, choices[begin:end].reshape(-1)).view(end - begin, cfg.negatives, dim)
        losses.append(_SampledNCE.apply(a[begin:end], pool[begin:end], neg, tau))
        weights.append(end - begin)
    return torch.sum(torch.stack([weight * loss for weight, loss in zip(weights, losses)])) / sum(weights)


def _pcl(states1: Tensor, states2: Tensor, cfg: Any, projector: nn.Module | None, generator: torch.Generator) -> Tensor:
    if states1.shape != states2.shape or states1.ndim != 4 or cfg.negatives < 1:
        raise ValueError('PCL requires matching [batch, windows, slots, features] states and positive negative count')
    if projector is not None:

        def project(x: Tensor) -> Tensor:
            if isinstance(projector, nn.ModuleList):
                if len(projector) != x.size(-2):
                    raise ValueError('PCL projector count differs from state slot count')
                return torch.stack([head(x.select(-2, slot)) for slot, head in enumerate(projector)], dim=-2)
            return projector(x)

        states1, states2 = project(states1), project(states2)
    batch, windows, slots, dim = states1.shape
    a = F.normalize(states1.reshape(-1, dim), p=2, dim=-1)
    b = F.normalize(states2.reshape(-1, dim), p=2, dim=-1)

    def direction(x: Tensor, y: Tensor) -> Tensor:
        if not cfg.same_slot or slots == 1:
            return _pcl_direction(x, y, cfg, windows * slots, generator)
        losses = [_pcl_direction(x[slot::slots], y[slot::slots], cfg, windows, generator) for slot in range(slots)]
        weights = torch.tensor([batch * windows] * slots, device=x.device, dtype=torch.float32)
        return (torch.stack(losses) * weights).sum() / weights.sum().clamp_min(1.0)

    return 0.5 * (direction(a, b) + direction(b, a))


class SSLLoss(nn.Module):
    """Own PCL sampling state while the caller supplies each global training step."""

    def __init__(self, sampling_seed: int | None = None) -> None:
        super().__init__()
        self.seed = random.randint(0, 2**31 - 1) if sampling_seed is None else int(sampling_seed)
        self._generators: dict[str, torch.Generator] = {}
        self._pending: dict[str, Tensor] = {}
        self.last_hgcl_rng: Tensor | None = None

    def get_extra_state(self) -> dict[str, Any]:
        """Include private sampler streams in training checkpoints."""
        states = {**self._pending, **{key: gen.get_state() for key, gen in self._generators.items()}}
        return dict(seed=self.seed, generators=states)

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Restore sampler streams lazily on their original devices."""
        self.seed = int(state['seed'])
        self._generators = {}
        self._pending = {key: value.cpu() for key, value in state['generators'].items()}

    def _generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if key not in self._generators:
            generator = torch.Generator(device=device).manual_seed(self.seed)
            if key in self._pending:
                generator.set_state(self._pending.pop(key))
            self._generators[key] = generator
        return self._generators[key]

    def forward(
        self, out: Any, cfg: Any, epoch: int, total_epochs: int, *, step: int
    ) -> tuple[Tensor, dict[str, float]]:
        """Return weighted terms with the reference's pinned temperatures."""
        if epoch < 0 or total_epochs < 1 or step < 0:
            raise ValueError('Expected nonnegative epoch/step and a positive total epoch count')
        loss_cfg = cfg.loss
        zero = torch.zeros((), device=out.proj_cls_1.device, requires_grad=True)
        hgcl = icl = pcl = zero
        if loss_cfg.hgcl_weight != 0:
            mask1 = out.mask_1[..., 1:] if out.mask_1 is not None else None
            mask2 = out.mask_2[..., 1:] if out.mask_2 is not None else None
            hgcl, self.last_hgcl_rng = _hgcl(out.proj_tokens_1, out.proj_tokens_2, mask1, mask2, loss_cfg.hgcl, step)
            hgcl = hgcl * loss_cfg.hgcl_weight
        if loss_cfg.icl_weight != 0:
            if out.proj_cls_1.shape != out.proj_cls_2.shape:
                raise ValueError('ICL requires matching CLS tensors')
            a = F.normalize(out.proj_cls_1, p=2, dim=-1, eps=1e-6)
            b = F.normalize(out.proj_cls_2, p=2, dim=-1, eps=1e-6)
            tau = torch.tensor(float(loss_cfg.icl_tau), device=a.device)
            targets = torch.arange(a.size(0), device=a.device)
            icl = (
                0.5
                * (F.cross_entropy(_logits(a @ b.T, tau), targets) + F.cross_entropy(_logits(b @ a.T, tau), targets))
                * loss_cfg.icl_weight
            )
        if loss_cfg.pcl_weight != 0 and out.pma_states_1 is not None and out.pma_states_2 is not None:
            pcl = (
                _pcl(
                    out.pma_states_1,
                    out.pma_states_2,
                    loss_cfg.pcl,
                    out.pcl_projector,
                    self._generator(out.pma_states_1.device),
                )
                * loss_cfg.pcl_weight
            )
        total = (hgcl + icl) + pcl
        if not torch.isfinite(total):
            raise ValueError('SSL loss is not finite')
        return total, {key: float(value.detach()) for key, value in dict(hgcl=hgcl, pcl=pcl, icl=icl).items()}


def ssl_loss(out: Any, cfg: Any, epoch: int, total_epochs: int, *, step: int) -> tuple[Tensor, dict[str, float]]:
    """Evaluate one fresh objective; training owns an SSLLoss to retain sampler state."""
    return SSLLoss()(out, cfg, epoch, total_epochs, step=step)
