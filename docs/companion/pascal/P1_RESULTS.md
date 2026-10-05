# Pascal P1 results — visual budget (development only)

**Label:** supplementary development analysis after the original final evaluation.
Does **not** change deployment selection or official holdout claims.

- Experiment / run revision: `cacbb90756de699eb4573a697487b4c4a9a82af1`
- GPU: `Tesla V100-SXM3-32GB` (34072559616 bytes); driver `570.124.06`
- Python `3.11.16`; torch `2.5.1+cu118` (CUDA 11.8); transformers `4.48.3`
- dtype / attention: `bfloat16` / `eager` (flash-attn disabled)
- dtype policy (as executed): bfloat16 after default `is_bf16_supported()` returned True. On V100 this reflects **tensor usability** (PyTorch default includes emulation); it does **not** establish native CC≥8 BF16 hardware acceleration. See `P1_RUNTIME_CLARIFICATION.md`. Float16 fallback was not applied.
- Slurm job: **117225** COMPLETED on `pascal-node10.l3s.intra`, elapsed ~00:12:30

## Compact table

| cap | n_req/ok/fail/miss | ANLS | EM | tile dist (actual) | mean tiles | gen-cap hits | gen tokens med | gen med/p95 s | req med/p95 s | peak alloc / reserved (request-scoped) |
|----:|-------------------:|-----:|---:|--------------------|-----------:|-------------:|---------------:|--------------:|--------------:|----------------------:|
| 12 | 208/208/0/0 | 0.8276 | 0.7644 | 7:13,10:3,13:192 | 12.582 | 0 | 6.0 | 1.748/1.960 | 1.972/2.305 | 3651364864/7130316800 |
| 4 | 208/208/0/0 | 0.7588 | 0.6538 | 3:40,5:168 | 4.615 | 0 | 6.0 | 0.700/0.919 | 0.863/1.159 | 2301478400/2464153600 |
| 1 | 208/208/0/0 | 0.5595 | 0.4183 | 1:208 | 1.000 | 0 | 6.0 | 0.287/0.446 | 0.382/0.665 | 1977783808/2109734912 |

## Paired quality vs fresh Pascal control (max_num=12)

- max_num=4: improved 18 / regressed 42 / tied 148; mean ΔANLS=-0.06884
- max_num=1: improved 7 / regressed 92 / tied 109; mean ΔANLS=-0.26815

## Historical Neumann base vs fresh Pascal control

- Exact response agreement: 206/208 (0.9904)
- Primary ANLS agreement: 207/208 (0.9952)
- Score changes: 1 (qid 44297 only; case-only text mismatch on qid 15330 keeps ANLS tied)
- Fresh control ANLS 0.8276 vs Neumann summary 0.8324 is accounted for by that single ANLS flip (1/208).
- Not a controlled hardware benchmark; do not rank Pascal vs Neumann latency.

## Main findings (conservative)

- Lowering only `max_num` reduces actual tile usage and cuts request/generation latency and **request-scoped** peak allocated memory on this V100 run.
- Development ANLS/EM fall as the budget shrinks (12→4→1): 0.8276/0.7644 → 0.7588/0.6538 → 0.5595/0.4183.
- Most paired changes vs control are regressions; a minority of improvements exist and should not be read as superiority of smaller budgets.
- Fresh original-cap predictions closely match historical Neumann development outputs under the same dtype policy.

## Figures

- `companion/pascal/results/figures/p1_anls_vs_request_latency.svg`
- `companion/pascal/results/figures/p1_anls_vs_tile_usage.svg`
- `companion/pascal/results/figures/p1_memory_summary.svg`

## Gallery

See `companion/pascal/results/p1_gallery.json` (images inspected; visual notes recorded there).

## Limitations

- Development-only; not holdout; no deployment reselection.
- Latency/memory are single-allocation workload observations; page cache can affect later conditions.
- Reserved memory includes allocator caching, not a per-request VRAM floor.
- Published memory columns are **request-scoped**. Raw field `gpu_memory_generation` is **not** independently generation-only under P1's reset monkey-patch (see `P1_RUNTIME_CLARIFICATION.md`).
- Tiny or mixed paired deltas are not statistical superiority claims.

## Resume

Compatible resume uses per-cap JSONL under `companion/pascal/artifacts/p1_runs/` with matching `run_fingerprint_sha256`.

