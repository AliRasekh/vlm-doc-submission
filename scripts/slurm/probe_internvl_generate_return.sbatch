#!/bin/bash
#SBATCH --job-name=vlm_doc_ivl_probe
#SBATCH --partition=ampere
#SBATCH --account=ampere
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/vlm_doc_internvl_genprobe_%j.out
#SBATCH --error=logs/vlm_doc_internvl_genprobe_%j.err

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
nvidia-smi || true

"$PYTHON" scripts/check_env_isolation.py
"$PYTHON" scripts/probe_internvl_generate_return.py \
  --config configs/models/internvl3_1b.yaml \
  --manifest data/manifests/docvqa_dev_v1.json \
  --question-ids 5158,292 \
  --out results/internvl_generate_return_probe.json \
  --cache-dir "$HF_HOME"

echo "INTERNVL_GENERATE_RETURN_PROBE_DONE"
