# vlm_doc — DocVQA evaluation & adaptation (submission snapshot)

Evaluate small open vision-language models on a frozen DocVQA design, select a
deployment candidate, compare a bounded language-attention LoRA adaptation, and
report sealed holdout results plus post-holdout supplementary analyses.

**Principal findings**

- Deployment candidate: **InternVL3-1B step 0** (development-selected).
- Holdout (`docvqa_holdout_v2`, n=504): base ANLS ≈ **0.7927**, LoRA v2 u200 ≈ **0.7993**;
  paired Δ ≈ +0.00664 with UCSF-cluster bootstrap 95% CI **[−0.00530, +0.01905]** (includes zero).
- Corrected LoRA under the fixed budget did **not** displace step 0 on development selection.
- Post-holdout Pascal supplements (tile budget, three-seed LoRA, visual-input dependence)
  are labeled supplementary and **did not** change sealed selection or holdout scores.

This repository is a **final submission snapshot**, not the full development Git history.
See [`SUBMISSION_PROVENANCE.md`](SUBMISSION_PROVENANCE.md).

## Primary report

- PDF: [`reports/vlm_doc_technical_report.pdf`](reports/vlm_doc_technical_report.pdf)
- Source / build: [`reports/vlm_doc_technical_report.tex`](reports/vlm_doc_technical_report.tex), [`reports/README.md`](reports/README.md)

## Environment setup

```bash
export PYTHONNOUSERSITE=1
conda create -y -n vlm_doc python=3.11
conda activate vlm_doc
pip install -r environment/requirements.txt
```

**Tested training/eval stack (recorded):** Python 3.11, PyTorch `2.5.1+cu118`,
transformers `4.48.3`, peft `0.14.0` (see `environment/versions.verified.txt`).
Generic `pip install` does **not** guarantee an identical CUDA wheel or GPU driver stack.

**Not included in Git (download separately):**

- Base InternVL3-1B / SmolVLM weights from Hugging Face (pinned revisions in configs).
- DocVQA page images and answer caches (use scripts + [`data/manifests/`](data/manifests/); see [`data/README.md`](data/README.md)).

Official adapter **is** included: `artifacts/internvl3_1b_lora_v2_u200/`.

```bash
sha256sum -c artifacts/internvl3_1b_lora_v2_u200_adapter_sha256.txt
```

## Inference example (image + question → answer)

Requires materialized images, answers, and base weights:

```bash
export PYTHONNOUSERSITE=1
python scripts/run_eval.py \
  --config configs/models/internvl3_1b.yaml \
  --manifest data/manifests/docvqa_dev_v1.json \
  --answers data/cache/docvqa_dev_v1_answers.json \
  --out-jsonl results/dev_internvl3_1b_run.jsonl \
  --summary-json results/dev_internvl3_1b_run_summary.json \
  --score
```

With the official adapter (construct pinned InternVL, then attach LoRA to `language_model`
via `load_language_lora` — do **not** trust PEFT `base_model_name_or_path`):

```bash
python scripts/run_eval.py \
  --config configs/models/internvl3_1b.yaml \
  --manifest data/manifests/docvqa_dev_v1.json \
  --answers data/cache/docvqa_dev_v1_answers.json \
  --out-jsonl results/dev_internvl_lora_run.jsonl \
  --summary-json results/dev_internvl_lora_run_summary.json \
  --lora-adapter artifacts/internvl3_1b_lora_v2_u200 \
  --train-update 200 \
  --score
```

Details: [`docs/report/SUBMISSION_GUIDE.md`](docs/report/SUBMISSION_GUIDE.md),
[`docs/report/adapter_metadata_audit.md`](docs/report/adapter_metadata_audit.md).

## Evaluation modes (keep these distinct)

| Mode | What it does | Entry |
|------|----------------|-------|
| **Display sealed tables** | Read compact JSON already in the repo | `results/holdout_final_table.json` |
| **Rescore saved predictions** | Recompute metrics from prediction JSONL (no model) | `scripts/analyze_holdout_final.py` (parameterized paths) |
| **Fresh model evaluation** | Run inference on GPU | `scripts/run_eval.py` |

Official sealed prediction filenames use the `results/holdout_*.jsonl` prefix.
Fresh runs may use other names; pass them explicitly to `analyze_holdout_final.py`.

## Training recipe

Bounded language-attention LoRA (InternVL3-1B): `configs/train/internvl3_1b_lora_v2.yaml`,
`scripts/train_internvl_lora.py`. Existing training logs are **not** overwritten unless
`--overwrite-existing-outputs` is set. Prefer a fresh `log_jsonl` / checkpoint directory.

Optional Slurm helpers live under `scripts/slurm/` and `companion/pascal/scripts/`
(cluster-specific; not required for understanding the method).

## Tests

```bash
export PYTHONNOUSERSITE=1
pip install pytest pillow  # if not already in the env
pytest
# Unit tests only (skip integration that need images/caches):
pytest -m "not integration"
```

Integration tests (`pytest.mark.integration`) need local DocVQA images or Pascal run
caches; they skip with an informative message when those are absent.

## Known limitations & future work

- Sampled DocVQA evaluation, not full official leaderboard parity.
- Unequal native visual-token budgets across model families.
- Possible pretrained exposure to DocVQA-related data is not ruled out from public cards alone.
- Server-GPU latency/memory do not transfer to edge devices without re-measurement.
- Sensible next steps: larger multi-seed budgets, matched visual preprocess across families,
  and targeted visual grounding diagnostics beyond whole-page replacement.

## Directory map

| Path | Role |
|------|------|
| `src/vlm_doc/` | Metrics, InternVL loading, LoRA helpers |
| `scripts/` | Eval, train, audits, holdout analysis |
| `configs/` | Pinned model + train YAMLs |
| `data/manifests/` | Frozen splits |
| `artifacts/internvl3_1b_lora_v2_u200/` | Official adapter |
| `results/` | Compact official + progressive tables |
| `reports/` | Integrated technical report |
| `companion/pascal/` | Post-holdout supplementary experiments |
| `docs/` | Protocols, progress notes, evidence index |
| `tests/` | Core unit tests |

## Supplementary Pascal results

Entry: [`companion/pascal/README.md`](companion/pascal/README.md).
Protocols/results: [`docs/companion/pascal/`](docs/companion/pascal/).
