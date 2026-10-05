# Report evidence index

Maps every reported number/figure in the Phase 4 technical-report package to its artifact, model/checkpoint, split, and code revision.

**Evaluation revision (protocol freeze before holdout):** `a9b0695ca05a593bdf680529bd673d28b0106e19` — see `results/phase04_evaluation_revision.json`.  
**Final packaging revision:** Git commit on `main` that introduces the Phase 4 compact evidence files listed below (holdout table/analysis/bootstrap + `docs/report/` + `docs/progress/phase04.md`); recorded in `results/phase04_packaging_revision.json` after that commit.

## Protocol / selection

| Claim | Artifact | Split / model | Revision |
|-------|----------|---------------|----------|
| Deployment candidate = InternVL step 0 | `docs/evaluation/final_protocol.md`, `results/dev_internvl_lora_v2_comparison.json` | development | Phase 3C / protocol freeze |
| Adaptation comparison = LoRA v2 u200 | `docs/evaluation/final_protocol.md`, adapter on disk | LoRA v2 u200 | Phase 3C / protocol freeze |
| Adapter SHA256 `3c2c34cd…c344180a` | `results/phase04_preflight.json`, checkpoint file | LoRA v2 u200 | protocol freeze |
| Holdout n=504 + manifest hashes | `results/phase04_preflight.json`, `data/manifests/docvqa_holdout_v2.json` | holdout_v2 | protocol freeze |

## Corrected training (v2)

| Claim | Artifact |
|-------|----------|
| Loss curve / first–last 20-update means (exact: 0.25614097751677034 / 0.21429072171449662) | `results/train_internvl3_1b_lora_v2_curves.json`, `docs/report/figures/train_loss_v2.svg`, `docs/examples/phase03c_train_loss_curve_v2.svg` |
| Gate termination check | `results/internvl_lora_v2_gate_ok.json` |
| Single-EOT supervised example | `results/train_lora_v2_supervised_example.json` |
| V1 double-EOT affectation 2000/2000 | `results/supervised_termination_audit_v1.json` |

## Robustness (development subset)

| Claim | Artifact |
|-------|----------|
| Base + v1 historical + v2 u200 table | `results/robustness_v2_analysis.json`, Phase 3B `results/robustness_v1/run_index.json` |

## Holdout (final)

| Claim | Artifact |
|-------|----------|
| Four-system primary/secondary scores | `results/holdout_final_table.json`, `results/holdout_final_analysis.json`, per-system sealed summaries below | holdout_v2 | job 84636 |
| SmolVLM-256M sealed summary | `results/holdout_smolvlm256m_summary.json` | holdout_v2 | job 84636 |
| SmolVLM-500M sealed summary | `results/holdout_smolvlm500m_summary.json` | holdout_v2 | job 84636 |
| InternVL3-1B sealed summary | `results/holdout_internvl3_1b_summary.json` | holdout_v2 | job 84636 |
| InternVL + LoRA v2 u200 sealed summary | `results/holdout_internvl3_1b_lora_v2_u200_summary.json` | holdout_v2 | job 84636 |
| Raw predictions | `results/holdout_*.jsonl` (local; not in Git) |
| Paired Δ + bootstrap CI | `results/holdout_internvl_anls_bootstrap.json`, `results/holdout_final_analysis.json` |
| Gallery | `docs/report/holdout_gallery.md`, `docs/report/gallery_examples/` |
| ANLS bar figure | `docs/report/figures/holdout_primary_anls.svg` |
| Corrected v2 train summary | `results/train_internvl3_1b_lora_v2_summary.json` |
| Corrected v2 development summaries | `results/dev_internvl3_1b_summary.json`, `results/dev_internvl3_1b_lora_v2_u100_summary.json`, `results/dev_internvl3_1b_lora_v2_u200_summary.json` |
| Completion validation (504 QID / hashes / GPU) | `results/phase04_completion_validation.json` |
| Report-input ZIP | `artifacts/vlm_doc_report_inputs.zip` (local; checksum `artifacts/vlm_doc_report_inputs_sha256.txt`) |

## Figures / diagrams

| Figure | Path | Generator / source |
|--------|------|--------------------|
| Holdout primary ANLS | `docs/report/figures/holdout_primary_anls.svg` | `scripts/analyze_holdout_final.py` |
| V2 train loss | `docs/examples/phase03c_train_loss_curve_v2.svg` | Phase 3C curves export |
| Architecture (LoRA on LM) | `docs/report/figures/architecture_lora_placement.svg` | static SVG |
| Pipeline flow | `docs/report/figures/pipeline_flow.svg` | static SVG |
| Robustness ANLS (dev subset, main) | `docs/report/figures/robustness_v2_anls.svg` | base + LoRA v2 u200 (`results/robustness_v2_analysis.json`) |
| Robustness ANLS (historical v1) | `docs/examples/robustness_v1_anls.svg` | Phase 3B; labeled historical |
| Train loss v2 (report copy) | `docs/report/figures/train_loss_v2.svg` | packaged curves |
| Paired adaptation CI | `docs/report/figures/paired_adaptation_ci.svg` | holdout bootstrap |
| Technical report PDF | `reports/vlm_doc_technical_report.pdf` | Phase 5 |

## Reproducibility commands (high level)

```bash
export PYTHONNOUSERSITE=1
# Env: ~/.conda/envs/vlm_doc with pinned requirements
# Data prep: scripts/prepare_frozen_splits.py / rebuild_holdout_v2.py / prepare_train_subset_v1.py (already frozen)
# Baseline eval: scripts/run_eval.py --config configs/models/<model>.yaml --manifest ... --score
# Corrected train: sbatch scripts/slurm/phase03c_internvl_lora_v2.sbatch
# Holdout: sbatch scripts/slurm/final_holdout_eval.sbatch
```

**Note:** Adapter-only checkpoints (`adapter_model.safetensors`) do **not** provide exact optimizer-state training resume.

## Limitations

See `docs/report/limitations.md`.
