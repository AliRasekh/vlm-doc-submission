# Submission / reviewer guide (Phase 5)

This private repository **is** the handoff. Share repository access with reviewers; do **not** require a separate delivery archive.

## What this repository contains

1. **Technical report PDF** — `reports/vlm_doc_technical_report.pdf` (sealed holdout + Pascal supplements)
2. **Code, configs, frozen manifests, compact results** — including `companion/pascal/`
3. **Official LoRA v2 u200 adapter** — tracked at `artifacts/internvl3_1b_lora_v2_u200/` (and optional zip `artifacts/internvl3_1b_lora_v2_u200_adapter.zip`)
4. **Checksum sidecar** — `artifacts/internvl3_1b_lora_v2_u200_adapter_sha256.txt`

**Still separate downloads (dependencies):** pinned InternVL3-1B base weights from Hugging Face and DocVQA images/answers when not already materialized locally.

| Item | Location |
|------|----------|
| Report PDF | `reports/vlm_doc_technical_report.pdf` |
| Report source / build | `reports/` (`README.md`) |
| Official adapter directory | `artifacts/internvl3_1b_lora_v2_u200/` |
| Adapter ZIP (same weights) | `artifacts/internvl3_1b_lora_v2_u200_adapter.zip` |
| Checksums | `artifacts/internvl3_1b_lora_v2_u200_adapter_sha256.txt` |
| Adapter metadata audit | `docs/report/adapter_metadata_audit.md` |
| Final protocol | `docs/evaluation/final_protocol.md` |
| Pascal companion | `companion/pascal/`, `docs/companion/pascal/` |

## Environment creation / setup

Run from the **repository root** unless noted.

```bash
export PYTHONNOUSERSITE=1
conda create -y -n vlm_doc python=3.11
conda activate vlm_doc
pip install -r environment/requirements.txt
# Pins: environment/versions.verified.txt (torch 2.5.1+cu118, transformers 4.48.3, peft 0.14.0, …)
```

Report-build tools stay separate from the ML env:

```bash
# repository root
export PYTHONNOUSERSITE=1
python3 -m venv .venv_report
.venv_report/bin/pip install cairosvg pillow
# System TeX: pdflatex, bibtex; optional mutool for page previews
```

## Data / image / answer preparation (frozen defaults)

From **repository root** (recreates frozen artifacts when needed):

```bash
export PYTHONNOUSERSITE=1
export PATH="$HOME/.conda/envs/vlm_doc/bin:$PATH"

python scripts/prepare_smoke_manifest.py \
  --dataset-id HuggingFaceM4/DocumentVQA \
  --dataset-revision a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4 \
  --k 8 \
  --manifest-out data/manifests/docvqa_smoke_v1.json \
  --answers-out data/cache/docvqa_smoke_v1_answers.json \
  --image-dir data/images/docvqa_smoke

python scripts/prepare_frozen_splits.py \
  --dataset-id HuggingFaceM4/DocumentVQA \
  --dataset-revision a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4 \
  --seed 42 \
  --dev-target 200 \
  --holdout-target 500 \
  --smoke-manifest data/manifests/docvqa_smoke_v1.json \
  --dev-manifest data/manifests/docvqa_dev_v1.json \
  --holdout-manifest data/manifests/docvqa_holdout_v1.json \
  --dev-answers data/cache/docvqa_dev_v1_answers.json \
  --holdout-answers data/cache/docvqa_holdout_v1_answers.json

python scripts/rebuild_holdout_v2.py

python scripts/prepare_train_subset_v1.py \
  --seed 42 \
  --target-n 2000 \
  --dataset-id HuggingFaceM4/DocumentVQA \
  --dataset-revision a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4 \
  --manifest-out data/manifests/docvqa_train_subset_v1.json \
  --answers-out data/cache/docvqa_train_subset_v1_answers.json \
  --image-dir data/images/docvqa_train_subset_v1

python scripts/prepare_robustness_subset_v1.py \
  --dev-manifest data/manifests/docvqa_dev_v1.json \
  --out data/manifests/docvqa_dev_robustness_v1.json \
  --seed 42 \
  --n-groups 16 \
  --max-per-group 4
```

## GPU inference / evaluation (Neumann compute allocation required)

`scripts/run_eval.py` requires CUDA. On Neumann, run these under a GPU allocation (interactive GPU session or Slurm), not on a CPU login node.

### Baseline evaluation (exact CLI)

From **repository root**:

```bash
export PYTHONNOUSERSITE=1
export PATH="$HOME/.conda/envs/vlm_doc/bin:$PATH"
python scripts/run_eval.py \
  --config configs/models/internvl3_1b.yaml \
  --manifest data/manifests/docvqa_dev_v1.json \
  --answers data/cache/docvqa_dev_v1_answers.json \
  --out-jsonl results/eval_internvl3_1b_dev.jsonl \
  --summary-json results/eval_internvl3_1b_dev_summary.json \
  --score
```

Holdout baseline (fresh evaluation example; use a dedicated output name):

```bash
python scripts/run_eval.py \
  --config configs/models/internvl3_1b.yaml \
  --manifest data/manifests/docvqa_holdout_v2.json \
  --answers data/cache/docvqa_holdout_v2_answers.json \
  --out-jsonl results/eval_internvl3_1b_holdout.jsonl \
  --summary-json results/eval_internvl3_1b_holdout_summary.json \
  --score
```

Official sealed prediction artifacts (if present locally) use the
`results/holdout_*.jsonl` / `results/holdout_*_summary.json` names for all four systems.
Compact sealed tables already in Git: `results/holdout_final_table.json`.

### Display sealed compact results (no model, no rescoring)

Open `results/holdout_final_table.json` and `results/holdout_final_analysis.json`.

### Rescore from saved prediction artifacts (no inference)

Requires local prediction JSONL + summaries + holdout answers cache. Write to a **fresh**
output directory so sealed files are not overwritten:

```bash
mkdir -p results/repro_holdout_analysis
python scripts/analyze_holdout_final.py \
  --smolvlm256m-jsonl results/holdout_smolvlm256m.jsonl \
  --smolvlm256m-summary results/holdout_smolvlm256m_summary.json \
  --smolvlm500m-jsonl results/holdout_smolvlm500m.jsonl \
  --smolvlm500m-summary results/holdout_smolvlm500m_summary.json \
  --internvl-jsonl results/holdout_internvl3_1b.jsonl \
  --internvl-summary results/holdout_internvl3_1b_summary.json \
  --internvl-lora-jsonl results/holdout_internvl3_1b_lora_v2_u200.jsonl \
  --internvl-lora-summary results/holdout_internvl3_1b_lora_v2_u200_summary.json \
  --lora-adapter-dir artifacts/internvl3_1b_lora_v2_u200 \
  --out results/repro_holdout_analysis/holdout_final_analysis.json \
  --table-json results/repro_holdout_analysis/holdout_final_table.json \
  --gallery-md results/repro_holdout_analysis/holdout_gallery.md
```

Prerequisites: `data/manifests/docvqa_holdout_v2.json`, `data/cache/docvqa_holdout_v2_answers.json`,
and the four prediction JSONL files (not shipped; regenerate only if authorized).
Also requires existing `results/holdout_internvl_anls_bootstrap.json` for the paired CI block.

### Fresh model evaluation (GPU; separate from sealed evidence)

```bash
python scripts/run_eval.py \
  --config configs/models/internvl3_1b.yaml \
  --manifest data/manifests/docvqa_holdout_v2.json \
  --answers data/cache/docvqa_holdout_v2_answers.json \
  --out-jsonl results/eval_internvl3_1b_holdout.jsonl \
  --summary-json results/eval_internvl3_1b_holdout_summary.json \
  --score
```

Do not point fresh eval outputs at the sealed `results/holdout_*.jsonl` paths.

## Adapter integrity and loading

### Verify checksums (repository root)

```bash
# cwd must be the repository root
sha256sum -c artifacts/internvl3_1b_lora_v2_u200_adapter_sha256.txt
```

Authoritative weights digest: `3c2c34cd751138df0a2a42bfebf90a9a9c7f64b89f2ebc8efe7f5517c344180a` for `artifacts/internvl3_1b_lora_v2_u200/adapter_model.safetensors`. The optional ZIP is a packaging convenience of the same bytes.

If you prefer the historical checkpoint path used in older scripts:

```bash
mkdir -p checkpoints/internvl3_1b_lora_v2/update_0200
unzip -o artifacts/internvl3_1b_lora_v2_u200_adapter.zip \
  -d checkpoints/internvl3_1b_lora_v2/update_0200
```

### Supported load route (required)

The loader **constructs** `OpenGVLab/InternVL3-1B` at revision `4415a3b810e636d11dfa86b0e9ba40bb00535aa8`, then attaches PEFT adapters to that model’s **`language_model`** submodule (`load_language_lora`). It does **not** auto-load `adapter_config.json`’s `base_model_name_or_path`.

That field currently reads `./pretrained/Qwen2.5-32B-Instruct` because it is **inherited** from InternVL3-1B’s embedded `llm_config._name_or_path` when PEFT saves the language-subtree adapter. It is stale upstream metadata, **not** evidence that a 32B model was trained. Do not replace it with the top-level VLM ID (the adapter wraps the language submodule). Generic PEFT “load base from adapter config” workflows are unsupported here.

Details: `docs/report/adapter_metadata_audit.md`.

### Exact eval with adapter (GPU allocation required)

```bash
export PYTHONNOUSERSITE=1
export PATH="$HOME/.conda/envs/vlm_doc/bin:$PATH"
python scripts/run_eval.py \
  --config configs/models/internvl3_1b.yaml \
  --manifest data/manifests/docvqa_holdout_v2.json \
  --answers data/cache/docvqa_holdout_v2_answers.json \
  --out-jsonl results/eval_internvl3_1b_lora_v2_u200_holdout.jsonl \
  --summary-json results/eval_internvl3_1b_lora_v2_u200_holdout_summary.json \
  --lora-adapter artifacts/internvl3_1b_lora_v2_u200 \
  --train-update 200 \
  --score
```

Equivalent historical path after unzip: `--lora-adapter checkpoints/internvl3_1b_lora_v2/update_0200`.

To reproduce the adapter instead of using the tracked weights: `sbatch scripts/slurm/phase03c_internvl_lora_v2.sbatch` (config `configs/train/internvl3_1b_lora_v2.yaml`).

## Report build

From the `reports/` directory:

```bash
cd reports
pdflatex -interaction=nonstopmode vlm_doc_technical_report.tex
bibtex vlm_doc_technical_report
pdflatex -interaction=nonstopmode vlm_doc_technical_report.tex
pdflatex -interaction=nonstopmode vlm_doc_technical_report.tex
```

Optional figure PDF refresh from SVG — run from **repository root** (paths are root-relative):

```bash
# repository root (not reports/)
.venv_report/bin/python - <<'PY'
from pathlib import Path
import cairosvg
for p in Path('docs/report/figures').glob('*.svg'):
    cairosvg.svg2pdf(url=str(p), write_to=str(Path('reports/figures')/(p.stem+'.pdf')))
PY
```

## Final protocol and results

- Protocol: `docs/evaluation/final_protocol.md` (freeze `a9b0695…`)
- Compact results: `results/holdout_final_table.json`, `results/holdout_final_analysis.json`, `results/holdout_internvl_anls_bootstrap.json`
- Report: `reports/vlm_doc_technical_report.pdf`
- Companion compact evidence: `companion/pascal/results/` (P1/P3/visual-dependence tables and figures)
- CPU close-out regressions (companion checkout with local artifacts):  
  `pytest companion/pascal/tests/test_p3_analyze.py companion/pascal/tests/test_p3_provenance.py companion/pascal/tests/test_visual_dep_analyze.py companion/pascal/tests/test_visual_dep_mapping.py companion/pascal/tests/test_p1_caps.py -q`
