# P3 verification record

## Scheduler / queue

| Job | Name | State | Elapsed | Node | Exit |
|-----|------|-------|---------|------|------|
| **117233** | vlm_p3_pre | COMPLETED | 00:15:42 | pascal-node10 | 0 |
| **117234** | vlm_p3_main | **TIMEOUT** | 04:00:22 | pascal-node10 | 0:0 (batch cancelled 0:15) |
| **117252** | vlm_p3_cont | **COMPLETED** | 02:11:23 | pascal-node09 | 0 |

Current `squeue` for user: **empty** (no active P3 servers/allocations).

## Git push (authorized)

| Check | Result |
|-------|--------|
| SSH auth | Success message; nonzero exit expected (no shell) |
| Branch | `companion/pascal-experiments` only; no force; no main merge |

## Compatibility gate — job **117233**

| Field | Value |
|-------|-------|
| GPU | Tesla V100-SXM3-32GB · CC 7.0 |
| BF16 default / native (`including_emulation=False`) | True / **False** |
| prefer_bf16 | retained (no FP16 switch) |
| Label mask / single EOT / grads / freeze / memorization / save-reload | all **ok** (logits max abs **0.0**) |
| Marker | `companion/pascal/artifacts/p3/gate_ok.json` |

## Completed adapters (exactly 200 updates)

| Seed | Job | Host / GPU | `adapter_model.safetensors` SHA256 | meta.update |
|-----:|-----|------------|-------------------------------------|------------:|
| 42 | 117234 | node10 / V100-SXM3-32GB | `fb35e2213a1a9f012cebcf5d4633138901718671b15750c31bc56a8c7f1b246e` | 200 |
| 43 | 117234 | node10 / V100-SXM3-32GB | `3b0d90463a3e0640e63e7588fbe66ffd972c83800c179d670b8adfaca2c42e2f` | 200 |
| 44 | 117252 | node09 / V100-PCIE-32GB | `9ffea211147bece915a22cd8136b6aee6d90938db3fe114aba79929dc2778847` | 200 |

Eval fingerprints record the same adapter weight hashes and attach via `load_language_lora` on the pinned InternVL3-1B (not PEFT’s stale `base_model_name_or_path`).

### Seed-44 restart

Interrupted 164-update attempt from 117234 preserved under  
`companion/pascal/artifacts/p3/interrupted/seed_44_job117234_20261004T055151Z/` and **excluded** from completed-run stats.  
Continuation cleared the working `seed_44/` tree and reran `scripts/train_internvl_lora.py` with seed=44 (fresh process → fresh base, adapters, optimizer, RNG). Order digest identical to interrupted (deterministic). **No optimizer-state resume.**

## Development evaluations (208 requested)

All four systems: **208 unique QIDs**, **0 duplicates**, **0 missing**, **0 failed**, **0 generation-cap hits**.  
Requested-set denominator used throughout; generation-cap hits (if any) remain in the means. Rescore matches written summaries (diff 0). Three-seed **0.0076** is sample SD (ddof=1), not a CI.

All evals executed on **pascal-node09 / V100-PCIE-32GB** under job **117252** (including fresh base and seeds 42–44 adapters).

## Protocol / source revisions

| Item | SHA |
|------|-----|
| Frozen protocol text | `9470055e3c0bfae06a24fad984a6640ddacd1ff0` |
| Job 117234 working tree | `90cdd12…` (protocol + wrappers; pre–walltime bump) |
| Job 117252 / evals | `54b92a47b5c03bb59da4b1d8ca4bdb69818ff33b` (16h continue launcher only) |
| Scientific settings change across these commits | **None** (seeds, caps, LR, LoRA, collator, manifests unchanged) |

## Protocol deviations (documented)

1. **Physical node/GPU change:** train seeds 42/43 on node10 **V100-SXM3-32GB**; train seed 44 + all evals on node09 **V100-PCIE-32GB**. Do not claim a single physical GPU for all runs.  
2. **Job timeout + full restart** of seed 44 (see above).  
3. **BF16** retained as usability/emulation on V100; not native CC≥8.

## Analysis verification

- Compact table `companion/pascal/results/p3_seed_table.json` agrees with independent rescore.
- Loss curves agree with per-update logs (first/last 20 means).
- Figures regenerated in maintenance: raw loss (low opacity) + trailing 20-update mean; ANLS plot with seed-44 label margin. PNGs written by analysis (not a separate visual QA pass beyond file generation).
- Fresh base ANLS/EM **exactly match** prior P1 cap-12 control means (0.827624 / 0.764423).

## Maintenance (2026-10-05)

- Coverage wording corrected to **80%** of the 2000-pool.
- Analyzer uses development-manifest requested QIDs as denominator; rejects duplicates/unexpected; no intersection primary metrics; fresh-base `n_ok`/`n_failed`/`n_missing` read from `scores.*`.
- `run_main` skip-existing validates coverage + fingerprints; incompatible caches preserved without overwrite/auto-rerun.
- Recomputed metrics from existing predictions: **unchanged** vs prior published table.

## Limitations remaining

- Three-seed descriptive stats only.
- Cross-node SXM3 vs PCIE hardware deviation.
- No holdout. No P4. No deployment seed selection.
- 1600 unique training QIDs ≈ 80% pool coverage per seed (shuffle+wrap).
