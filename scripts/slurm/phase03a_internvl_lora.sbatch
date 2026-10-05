#!/bin/bash
#SBATCH --job-name=vlm_doc_lora3a
#SBATCH --partition=ampere
#SBATCH --account=ampere
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/vlm_doc_lora3a_%j.out
#SBATCH --error=logs/vlm_doc_lora3a_%j.err

# Phase 3A: correctness gate -> 200-update LoRA train -> dev eval 100/200 -> compare
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results checkpoints/internvl3_1b_lora_v1

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
unset TRANSFORMERS_CACHE || true
export TORCH_HOME="${TORCH_HOME:-$PWD/.torch_cache}"
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME"

if [ -d /local ] && [ -w /local ]; then
  export TMPDIR="/local/${USER}/vlm_doc/tmp"
  mkdir -p "$TMPDIR"
fi

ENV_PREFIX="${VLM_DOC_ENV:-$HOME/.conda/envs/vlm_doc}"
PYTHON="${ENV_PREFIX}/bin/python"
echo "host=$(hostname) date=$(date -Is) SLURM_JOB_ID=${SLURM_JOB_ID:-}"
nvidia-smi || true
"$PYTHON" scripts/check_env_isolation.py
"$PYTHON" - <<'PY'
import peft, torch, transformers
print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__)
assert torch.__version__.startswith("2.5.1")
assert transformers.__version__ == "4.48.3"
assert peft.__version__ == "0.14.0"
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

# Require image integrity audit artifact (CPU job 82979 or prior).
if [ ! -f results/train_subset_v1_image_integrity_audit.json ]; then
  echo "ERROR: missing image integrity audit; run CPU audit first" >&2
  exit 2
fi
"$PYTHON" - <<'PY'
import json
r=json.load(open("results/train_subset_v1_image_integrity_audit.json"))
assert r.get("ok") is True, r
print("IMAGE_AUDIT_OK repairs", r.get("n_repairs"), "examples_sha", r.get("examples_content_sha256")[:16])
PY

echo "=== CORRECTNESS GATE ==="
"$PYTHON" scripts/gate_internvl_lora.py \
  --config configs/train/internvl3_1b_lora_v1.yaml \
  --cache-dir "$HF_HOME" \
  --out results/internvl_lora_gate_ok.json

echo "=== MAIN TRAIN 200 UPDATES ==="
"$PYTHON" scripts/train_internvl_lora.py \
  --config configs/train/internvl3_1b_lora_v1.yaml \
  --cache-dir "$HF_HOME"

echo "=== DEV EVAL STEP 100 ==="
"$PYTHON" scripts/run_eval.py \
  --config configs/models/internvl3_1b.yaml \
  --manifest data/manifests/docvqa_dev_v1.json \
  --answers data/cache/docvqa_dev_v1_answers.json \
  --out-jsonl results/dev_internvl3_1b_lora_u100.jsonl \
  --summary-json results/dev_internvl3_1b_lora_u100_summary.json \
  --cache-dir "$HF_HOME" \
  --lora-adapter checkpoints/internvl3_1b_lora_v1/update_0100 \
  --train-update 100 \
  --score

echo "=== DEV EVAL STEP 200 ==="
"$PYTHON" scripts/run_eval.py \
  --config configs/models/internvl3_1b.yaml \
  --manifest data/manifests/docvqa_dev_v1.json \
  --answers data/cache/docvqa_dev_v1_answers.json \
  --out-jsonl results/dev_internvl3_1b_lora_u200.jsonl \
  --summary-json results/dev_internvl3_1b_lora_u200_summary.json \
  --cache-dir "$HF_HOME" \
  --lora-adapter checkpoints/internvl3_1b_lora_v1/update_0200 \
  --train-update 200 \
  --score

echo "=== COMPARE ==="
"$PYTHON" scripts/compare_internvl_lora_dev.py \
  --baseline results/dev_internvl3_1b.jsonl \
  --step100 results/dev_internvl3_1b_lora_u100.jsonl \
  --step200 results/dev_internvl3_1b_lora_u200.jsonl \
  --out results/dev_internvl_lora_v1_comparison.json \
  --examples-md docs/examples/lora_v1_before_after.md

echo "PHASE3A_PIPELINE_DONE"
