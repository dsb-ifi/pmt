# Matched Table-4 cost microbenchmark — GH200

**Status:** COMPLETE

## Environment

- Timestamp: `2026-11-01T23:09:52+01:00`
- GPU: `NVIDIA GH200 120GB` (97871 MiB); torch sees `NVIDIA GH200 120GB`
- NVIDIA driver: `580.65.06`; nvidia-smi CUDA compatibility: `13.0`
- PyTorch: `2.10.0+cu126`; PyTorch CUDA build: `12.6`
- Python: `3.11.17`; host: `gh200-node`
- Git: `b3166c9633ec5be758d2ba2f9ae90e9bff1bb872`; dirty: **true** (1 porcelain entries)
- Manifest submission commit: `null`

## Controlled setup

Each cell uses the seed-42 manifest override string and its recorded profile, resolved with `PMTConfig.load_config(..., lightweight=False)`. Training uses batch 256, BF16 autocast, production fused Adam parameter groups and gradient clipping, 10 warmup steps, and 50 synchronized timed full optimizer steps.
Inputs are two fixed synthetic FP32 tensors of shape `[B,C,L]`, allocated before timing. The full production SSL wrapper computes the configured HGCL+ICL stack and PMT PCL where active. No dataloader, augmentation, W&B, scheduler step, checkpoint, or probe is in the timed path.
Inference is the canonical single-view `forward_backbone_output` path in FP32 under `model.eval()` and `torch.no_grad()`, with 10 warmup and 50 synchronized timed batches.
- Resolved compile controls: `[{'compile_backbone': False, 'compile_encoder': False, 'hgcl_compile': False, 'pma_compile_mode': 'reduce-overhead', 'pma_compile_per_window': True, 'pma_shared_compile': False}]`

## Results

| Dataset | Arm | Batch | Params (M) | Train ms/step median [Q1–Q3] | Peak train GB | Inference seq/s (median) | Inference ms/batch |
|---|---|---:|---:|---:|---:|---:|---:|
| HAR | Transformer-XL adapter | 256 | 18.350 | 206.42 [205.41–209.40] | 18.52 | 7265 | 35.24 |

Peak training memory is CUDA `max_memory_allocated` divided by 10^9 bytes and includes the model, two synthetic inputs, gradients, optimizer state, and active loss graph.

## Derived step-time ratios

| Dataset | PMT / Transformer-XL | xLSTM / PMT |
|---|---:|---:|
| HAR | — | — |

Ratios use same-batch per-cell median full-step times. A value above 1 means the numerator is slower; cross-batch OOM retries are not converted into ratios.

## Resolved geometry

| Dataset | Profile | C | L | d_model | Tokenizer | Patch / stride | Resolved run batch | Adapter blocks (PMT / xLSTM / TXL) |
|---|---|---:|---:|---:|---|---:|---:|---:|
| HAR | neurips | 9 | 128 | 320 | pmt | 3 / 1 | 256 | 9 / 6 / 9 |

## Parity checks and deviations

- PASS: within every dataset, all three measured arms share C, L, d_model, tokenizer type, patch/stride, and input-stem settings.
- **Batch-size conflict:** the exact epilepsy manifest resolves training batch 512, while the benchmark prompt mandates batch 256. Epilepsy is therefore measured at 256; all other source rows resolve to 256.
- Precision is launch/runtime-resolved rather than persisted in PMTConfig. In this allocation `pick_mixed_dtype` selected BF16; inference follows the repository's FP32 frozen-probe path.
- The manifest was submitted at a different project commit than the measured checkout; both SHAs and the dirty flag are reported above. The pre-existing dirty changes were not modified.
- The resolved local-loss selector is `hgcl`, so the production wrapper uses base `HierarchicalGaussianContrastiveLoss`; the visible `objective_variant=weighted_multipos` field is not active for this selector.
- All resolved compile switches are off, so the 10 training warmups initialize kernels and Adam state but do not trigger `torch.compile`.
- No measured cell required an OOM retry from batch 256.

## Sources

- Benchmark script: a GPU microbenchmark script (not included)
