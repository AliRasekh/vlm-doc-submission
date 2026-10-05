# `src/vlm_doc`

Reusable library code for loading SmolVLM/Idefics3-style models, running single-example greedy inference, scoring, and resume fingerprints.

## Modules

| Module | Purpose |
|--------|---------|
| `model.py` | SmolVLM load + unique parameter count (&lt;1e9 assert) |
| `infer.py` | SmolVLM greedy generate path |
| `adapters/internvl.py` | InternVL3 dynamic preprocess + `generate()`-scoped inference |
| `metrics.py` | Primary `anls_normalized_strict_v2`; legacy/compat variants |
| `run_fingerprint.py` | Manifest content hash; `inference_v2` fingerprint |
| `metadata.py` | Git commit/dirty + Slurm env metadata |

## Why references stay out of inputs

DocVQA measures whether the model can read the **page image** and answer the question. Feeding gold answers into the prompt would invalidate the evaluation. Reference answers are loaded only for offline scoring from gitignored sidecars.

## Fingerprints

Inference fingerprint hashes model/revision, manifest **content** hash, instruction, generation settings, config, resolved dtype, processor settings, software versions, and inference source files. Documentation-only commits do not change it. Scoring provenance is separate so metric-only changes can CPU-rescore existing predictions.

## GPU memory fields

`gpu_memory.peak_allocated_bytes` / `peak_reserved_bytes` come from PyTorch’s caching allocator around each `generate()` call after `reset_peak_memory_stats`. Convert to GiB with `bytes / 1024**3`. They are **not** host MaxRSS and **not** full `nvidia-smi` used memory.
