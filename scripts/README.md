# Scripts

| Script | Role |
|--------|------|
| `download_model.py` | Prefetch pinned SmolVLM files into `.hf_cache` |
| `prepare_smoke_manifest.py` | Phase 1 8-example smoke set |
| `prepare_frozen_splits.py` | Phase 2A full-val → ~200 dev + ~500 holdout (historical `doc_id` grouping) |
| `rebuild_holdout_v2.py` | UCSF-grouped holdout v2; keeps fixed development IDs; seed 42 |
| `audit_split_leakage.py` | Fail on UCSF/page/hash overlap between holdout and development/smoke |
| `rescore_predictions.py` | CPU-only rescore under a selected metric version |
| `compare_dev_three_models.py` | Three-model development comparison (strict v2 + diagnostic EM) |
| `export_full_review_packet.py` | Concatenate actual source/results into a full review packet |
| `probe_internvl_generate_return.py` | Bounded InternVL generate-return probe (adapter vs official chat) |
| `prepare_train_subset_v1.py` | Leakage-checked ~2k train subset from pinned DocVQA train |
| `analyze_internvl_error_sample.py` | Seeded ≤20 non-perfect InternVL development error sample |
| `audit_train_subset_images_v1.py` | Verify/repair train PNG RGB provenance vs frozen manifest |
| `gate_internvl_lora.py` | Phase 3A correctness gate before main LoRA run |
| `train_internvl_lora.py` | Bounded InternVL language_model LoRA training |
| `compare_internvl_lora_dev.py` | Dev ANLS selection + before/after examples |
| `audit_phase03a_evidence.py` | Re-verify Phase 3A train/eval evidence from artifacts |
| `audit_supervised_termination_v1.py` | Count v1 double-EOT affectation without image materialization |
| `prepare_robustness_subset_v1.py` | Deterministic 16-UCSF-group development robustness subset |
| `run_robustness_dev_v1.py` | InternVL clean/half/jpeg robustness evals (v1/v2 adapters) |
| `analyze_robustness_v1.py` | Robustness tables, fresh-vs-historical, gallery |
| `bootstrap_holdout_anls_diff.py` | Prepared UCSF-cluster bootstrap (holdout; do not run until authorized) |
| `analyze_phase02b_cpu_followup.py` | EM formatting diagnostic + gallery provenance (CPU) |
| `check_env_isolation.py` | Fail-fast if `PYTHONNOUSERSITE` was not set by the launcher |
| `smoke_inference.py` | Legacy smoke runner (Phase 1 reproducibility) |
| `run_eval.py` | General eval entrypoint (config + manifest + optional `--score`) |

## Eval command (GPU via Slurm)

```bash
export PYTHONNOUSERSITE=1   # must be set BEFORE python starts
python scripts/run_eval.py \
  --config configs/models/smolvlm_500m.yaml \
  --manifest data/manifests/docvqa_dev_v1.json \
  --answers data/cache/docvqa_dev_v1_answers.json \
  --out-jsonl results/dev_smolvlm500m.jsonl \
  --summary-json results/dev_smolvlm500m_summary.json \
  --score
```

Resume only reuses JSONL rows whose `inference_v2` fingerprint matches. Mixed-fingerprint files are rejected (use a new `--out-jsonl`).
