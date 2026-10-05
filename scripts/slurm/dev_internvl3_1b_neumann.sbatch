#!/bin/bash
#SBATCH --job-name=vlm_doc_dev_ivl
#SBATCH --partition=ampere
#SBATCH --account=ampere
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/vlm_doc_dev_internvl_%j.out
#SBATCH --error=logs/vlm_doc_dev_internvl_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results

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
if [ ! -x "$PYTHON" ]; then
  echo "ERROR: missing python at $PYTHON" >&2
  exit 2
fi

echo "host=$(hostname)"
echo "date=$(date -Is)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PYTHONNOUSERSITE=$PYTHONNOUSERSITE"
nvidia-smi || true

"$PYTHON" scripts/check_env_isolation.py
"$PYTHON" - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("bf16", torch.cuda.is_bf16_supported())
PY

# Smoke on first 4 development examples, then full set if smoke succeeds.
"$PYTHON" scripts/run_eval.py \
  --config configs/models/internvl3_1b.yaml \
  --manifest data/manifests/docvqa_dev_v1.json \
  --answers data/cache/docvqa_dev_v1_answers.json \
  --out-jsonl results/dev_internvl3_1b_smoke4.jsonl \
  --summary-json results/dev_internvl3_1b_smoke4_summary.json \
  --cache-dir "$HF_HOME" \
  --limit 4 \
  --score

"$PYTHON" scripts/run_eval.py \
  --config configs/models/internvl3_1b.yaml \
  --manifest data/manifests/docvqa_dev_v1.json \
  --answers data/cache/docvqa_dev_v1_answers.json \
  --out-jsonl results/dev_internvl3_1b.jsonl \
  --summary-json results/dev_internvl3_1b_summary.json \
  --cache-dir "$HF_HOME" \
  --score

echo "DEV_INTERNVL3_1B_BASELINE_DONE"
