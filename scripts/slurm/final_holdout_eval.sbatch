#!/bin/bash
#SBATCH --job-name=vlm_doc_holdout
#SBATCH --partition=ampere
#SBATCH --account=ampere
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/vlm_doc_holdout_%j.out
#SBATCH --error=logs/vlm_doc_holdout_%j.err

# Phase 4 final holdout: four systems sequential on one A100.
# Adaptation comparison = LoRA v2 update 200 only (never v1 / never v2 u100).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results docs/report/figures

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
unset TRANSFORMERS_CACHE || true
export TORCH_HOME="${TORCH_HOME:-$PWD/.torch_cache}"

ENV_PREFIX="${VLM_DOC_ENV:-$HOME/.conda/envs/vlm_doc}"
PYTHON="${ENV_PREFIX}/bin/python"
echo "host=$(hostname) date=$(date -Is) SLURM_JOB_ID=${SLURM_JOB_ID:-}"
nvidia-smi || true
"$PYTHON" scripts/check_env_isolation.py
"$PYTHON" - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__)
assert torch.__version__.startswith("2.5.1")
assert transformers.__version__ == "4.48.3"
assert peft.__version__ == "0.14.0"
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY

# Preflight artifact required
"$PYTHON" - <<'PY'
import json, hashlib
from pathlib import Path
p=json.load(open("results/phase04_preflight.json"))
assert p.get("ok") is True, p
exp="3c2c34cd751138df0a2a42bfebf90a9a9c7f64b89f2ebc8efe7f5517c344180a"
h=hashlib.sha256(Path("checkpoints/internvl3_1b_lora_v2/update_0200/adapter_model.safetensors").read_bytes()).hexdigest()
assert h==exp, (h, exp)
print("PREFLIGHT_AND_ADAPTER_OK", h[:16])
PY

# Bootstrap multiplicity self-test
"$PYTHON" scripts/bootstrap_holdout_anls_diff.py --self-test-multiplicity

MANIFEST=data/manifests/docvqa_holdout_v2.json
ANSWERS=data/cache/docvqa_holdout_v2_answers.json

echo "=== SmolVLM-256M ==="
"$PYTHON" scripts/run_eval.py \
  --config configs/models/smolvlm_256m.yaml \
  --manifest "$MANIFEST" \
  --answers "$ANSWERS" \
  --out-jsonl results/holdout_smolvlm256m.jsonl \
  --summary-json results/holdout_smolvlm256m_summary.json \
  --cache-dir "$HF_HOME" \
  --score

echo "=== SmolVLM-500M ==="
"$PYTHON" scripts/run_eval.py \
  --config configs/models/smolvlm_500m.yaml \
  --manifest "$MANIFEST" \
  --answers "$ANSWERS" \
  --out-jsonl results/holdout_smolvlm500m.jsonl \
  --summary-json results/holdout_smolvlm500m_summary.json \
  --cache-dir "$HF_HOME" \
  --score

echo "=== InternVL3-1B base (deployment candidate) ==="
"$PYTHON" scripts/run_eval.py \
  --config configs/models/internvl3_1b.yaml \
  --manifest "$MANIFEST" \
  --answers "$ANSWERS" \
  --out-jsonl results/holdout_internvl3_1b.jsonl \
  --summary-json results/holdout_internvl3_1b_summary.json \
  --cache-dir "$HF_HOME" \
  --score

echo "=== InternVL3-1B LoRA v2 u200 (adaptation comparison) ==="
"$PYTHON" scripts/run_eval.py \
  --config configs/models/internvl3_1b.yaml \
  --manifest "$MANIFEST" \
  --answers "$ANSWERS" \
  --out-jsonl results/holdout_internvl3_1b_lora_v2_u200.jsonl \
  --summary-json results/holdout_internvl3_1b_lora_v2_u200_summary.json \
  --cache-dir "$HF_HOME" \
  --lora-adapter checkpoints/internvl3_1b_lora_v2/update_0200 \
  --train-update 200 \
  --score

echo "=== Bootstrap InternVL ANLS diff (base vs v2 u200) ==="
"$PYTHON" scripts/bootstrap_holdout_anls_diff.py \
  --manifest "$MANIFEST" \
  --base-jsonl results/holdout_internvl3_1b.jsonl \
  --adapted-jsonl results/holdout_internvl3_1b_lora_v2_u200.jsonl \
  --replicates 2000 \
  --seed 42 \
  --out results/holdout_internvl_anls_bootstrap.json

echo "=== Final analysis ==="
"$PYTHON" scripts/analyze_holdout_final.py \
  --out results/holdout_final_analysis.json \
  --table-json results/holdout_final_table.json \
  --gallery-md docs/report/holdout_gallery.md

echo "HOLDOUT_PROTOCOL_DONE"
