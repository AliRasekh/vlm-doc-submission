# Phase 4 — Final holdout evaluation and report evidence

**Dates:** 2026-10-03  
**Host:** Neumann  
**Status:** COMPLETED (inference + analysis + compact evidence packaging)  
**Evaluation revision (protocol freeze):** `a9b0695ca05a593bdf680529bd673d28b0106e19`  
**Holdout job:** **84636** COMPLETED ExitCode `0:0`, 00:26:26 on `gpunode02` / **NVIDIA A100-PCIE-40GB**  
**Launcher:** `scripts/slurm/final_holdout_eval.sbatch`  
**Eval code in fingerprints:** `scripts/run_eval.py` SHA256 `a36f89f42744c4eb5d5ef160aa4165d0a05e618d1fae6b129ba4eb212b7c2a58`  
**Completion validation:** `results/phase04_completion_validation.json` (504 unique OK/system; adapter hash match)

## Preflight

`results/phase04_preflight.json` — **ok**

- Collator termination tests 11/11; gate `termination_single_eos` ok.
- V2 supervised example / qid 46350: exactly one EOT; no supervised trailing newline.
- Main v2 train was a separate process after gate (job 83333 log).
- Adapter v2 u200 SHA256 matches disk `3c2c34cd…c344180a`.
- Holdout manifest 504 unique QIDs; 504/504 images hash-verified.
- Current train/dev/holdout_v2: zero UCSF/page/QID/image-hash overlaps.
- Launcher resolves **only** v2 u200 (never v1 / never u100).

### V2 loss windows (not comparable to v1 label objective)

| Statistic | Value |
|-----------|------:|
| Updates | 200 |
| First update mean loss | 0.2532 |
| Final update mean loss | 0.3007 |
| Mean first 20 updates | 0.25614097751677034 |
| Mean last 20 updates | 0.21429072171449662 |

Semantics: per-optimizer-update mean of `grad_accum=8` microbatch losses.

## Frozen systems

| Tag | Identity |
|-----|----------|
| Deployment candidate | InternVL3-1B step 0 @ `4415a3b8…` |
| Adaptation comparison | LoRA v2 update 200 (`3c2c34cd…c344180a`) |
| Also evaluated | SmolVLM-256M @ `7e3e67ed…`; SmolVLM-500M @ `a7da5b98…` |

Holdout rankings were **not** used to reselect checkpoints.

## Holdout primary results (n=504 requested-set)

Primary metric: `anls_normalized_strict_v2`. Sources: `results/holdout_final_table.json`, per-system summaries.

| System | Params | ANLS | EM | Sec. ANLS (vlk) | Diag EM | Cap | Med / p95 gen-s | Peak alloc GiB |
|--------|-------:|-----:|---:|----------------:|--------:|----:|----------------:|---------------:|
| SmolVLM-256M | 256.5M | 0.5753 | 0.3234 | 0.5842 | 0.5159 | 6 | 0.214 / 0.410 | 2.78 |
| SmolVLM-500M | 507.5M | 0.5997 | 0.0694 | 0.6305 | 0.6052 | 0 | 0.244 / 0.427 | 3.26 |
| InternVL3-1B (step 0) | 938.2M | 0.7927 | 0.7282 | 0.7967 | 0.7361 | 0 | 0.482 / 0.647 | 3.42 |
| InternVL + LoRA v2 u200 | 938.2M (+adapters) | 0.7993 | 0.7401 | 0.8023 | 0.7500 | 0 | 0.538 / 0.799 | 3.43 |

All four systems: `n_ok=504`, `n_failed=0`, `n_missing=0`. Latency = generation-only, CUDA-synchronized, warm-up excluded. GPU memory = max over per-example `torch.cuda.max_memory_{allocated,reserved}` (not nvidia-smi). Hardware: **NVIDIA A100-PCIE-40GB**.

### Paired InternVL base vs LoRA v2 u200

| Quantity | Value |
|----------|------:|
| Improved / regressed / tied | 13 / 11 / 480 |
| Mean Δ ANLS (adapted − base; unrounded) | +0.006639 |
| Bootstrap mean Δ (B=2000, seed 42) | +0.00674 |
| Percentile 95% CI | [−0.00530, +0.01905] |

Bootstrap: UCSF-cluster resample with replacement, multiplicity retained (self-test ok), question-weighted means (`results/holdout_internvl_anls_bootstrap.json`).  
**Interpretation:** Positive holdout point estimate for adaptation; the 95% CI **includes zero**, which does **not** demonstrate equivalence. An interval excluding zero would still not establish broad superiority beyond this evaluation design. Deployment candidate remains **InternVL step 0**.

## Artifacts

| Path | Role |
|------|------|
| `results/holdout_*_{summary,jsonl}` | Per-system scores + predictions (jsonl local/gitignored) |
| `results/holdout_final_table.json` | Compact four-system table |
| `results/holdout_final_analysis.json` | Paired deltas, gallery selection, caveats |
| `results/holdout_internvl_anls_bootstrap.json` | Bootstrap CI |
| `docs/report/figures/*` | ANLS bar + architecture + pipeline diagrams |
| `docs/report/holdout_gallery.md` + `gallery_examples/` | Deterministic examples + thumbs (no visual-cause claims) |
| `docs/report/evidence_index.md` | Claim → artifact map |
| `docs/report/limitations.md` | Explicit limitations |
| `results/phase04_review.txt` | Full review packet (local/gitignored) |
| `artifacts/vlm_doc_report_inputs.zip` | Compact report-input bundle (local; SHA in `artifacts/vlm_doc_report_inputs_sha256.txt`) |

## Interpretation (bounded)

- Development-selected **InternVL step 0** remains the deployment candidate; holdout does not authorize reselection.
- LoRA v2 u200 shows a small positive holdout ANLS point estimate vs base; the UCSF-cluster 95% CI includes zero.
- Cross-family gaps are **configured-system** comparisons (native visual-token budgets differ).
- Diagnostic terminal-period EM remains secondary (formatting sensitivity), not official accuracy.
- Development-only robustness (Phase 3B/3C) is separate from holdout evaluation.

## Non-actions

No further training, prompt/metric tuning, checkpoint reselection from holdout, email/publish, or repo-visibility changes in this phase.
