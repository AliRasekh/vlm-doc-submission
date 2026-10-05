# Phase 3B — Development robustness + final-protocol preparation

**Dates:** 2026-10-02  
**Host:** Neumann  
**Status:** COMPLETED preparation + robustness experiment (holdout sealed; no new training)

## Selection freeze (from Phase 3A)

| Role | Choice |
|------|--------|
| Deployment candidate | Original InternVL3-1B (**step 0**) |
| Best trained checkpoint | LoRA **update 100** |
| Update 200 | Retained experiment result; **not selected** |
| Fine-tuning vs primary dev ANLS | **Did not improve** |

No further training, prompt changes, scoring changes, or hyperparameter search in this phase. Holdout remains sealed pending independent train-source review.

## Phase 3A evidence re-verification

Artifact: `results/phase03a_evidence_audit.json` (`ok: true`)

- 200 optimizer updates; 1600 sample presentations; 1600 unique QIDs.
- Logged `loss` = **mean microbatch loss within each update’s grad_accum window (8)** — not a single-example loss and not a sliding average across updates.
- Mean loss first 20 updates ≈ 0.3288; last 20 ≈ 0.1583 (training statistics on potentially different examples).
- Loss curve: `docs/examples/phase03a_train_loss_curve.svg`
- Diagnostic gate adapters/optimizer discarded (`TemporaryDirectory` rmtree); main train is a separate process after `GATE_OK`.
- u100/u200 eval fingerprints match on-disk adapter SHA256s; all 208 predictions accounted (`ok`) at steps 0/100/200.
- Recomputed 4 improved / 4 regressed / 200 tied; eight examples: `docs/examples/phase03a_eight_changed.md`.

Observations only: train loss fell while primary development ANLS did not improve. This is **not** treated as established overfitting or catastrophic forgetting.

## Robustness subset

Manifest: `data/manifests/docvqa_dev_robustness_v1.json`  
- Seed 42; 16 UCSF groups sampled without replacement from sorted group IDs.  
- Up to 4 questions per group via seeded within-group shuffle.  
- **54 questions / 16 groups** (some groups have &lt;4 questions).  
- Selection independent of predictions/scores.  
- `manifest_sha256` recorded in the manifest file.

## Perturbations (synthetic, limited)

Applied independently to clean RGB copies (originals immutable):

1. **clean** — identity  
2. **half_detail** — resize to floor(w/2), floor(h/2) (min 1) with **BICUBIC**, restore exact original size  
3. **jpeg_q40** — JPEG encode quality=40, subsampling=2 (4:2:0), decode RGB  

Pre-model dimensions unchanged; InternVL tiling settings unchanged. Tile counts asserted identical across conditions per image. Fingerprints include perturbation identity + adapter identity (no clean→corrupt resume).

## GPU job

| Job | Role | State | GPU | Elapsed |
|-----|------|-------|-----|---------|
| **83083** | Six robustness evals + analysis | COMPLETED 0:0 | gpunode03 / **A100-PCIE-40GB** | **00:08:29** |

## Results (primary ANLS strict v2; n=54; 0 failures/missing; 0 gen-cap hits)

| Model | Condition | ANLS | EM | Δ vs own clean |
|-------|-----------|-----:|---:|---------------:|
| base (step 0) | clean | 0.7671 | 0.7037 | 0 |
| base | half_detail | 0.7691 | 0.7222 | +0.0021 |
| base | jpeg_q40 | 0.7292 | 0.6667 | −0.0379 |
| LoRA u100 | clean | 0.7535 | 0.7222 | 0 |
| LoRA u100 | half_detail | 0.7700 | 0.7222 | +0.0165 |
| LoRA u100 | jpeg_q40 | 0.7144 | 0.6667 | −0.0391 |

Paired comparisons on identical questions. Tiny deltas (e.g. half_detail vs clean) are **not** statistical superiority on this small subset.

Fresh clean vs historical full-dev predictions on subset QIDs: **54/54 exact match** for both base and LoRA u100 (no systematic discrepancy).

Artifacts: `results/robustness_v1_analysis.json`, `docs/examples/robustness_v1_gallery.md`, `docs/examples/robustness_v1_anls.svg`.

## Final holdout protocol (prepared, not executed)

- Spec: `docs/evaluation/final_protocol.md`  
- Launcher: `scripts/slurm/final_holdout_eval.sbatch` (**do not submit in 3B**)  
- Bootstrap helper: `scripts/bootstrap_holdout_anls_diff.py` (2000 UCSF-cluster replicates, seed 42)

**No holdout inference in Phase 3B. No additional training.**
