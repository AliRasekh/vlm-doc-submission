#!/bin/bash
# Visual-input dependence: three conditions on InternVL3-1B base (dev only).
#SBATCH --job-name=vlm_visdep
#SBATCH --partition=volta
#SBATCH --account=pascal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=companion/pascal/logs/visual_dep_%j.out
#SBATCH --error=companion/pascal/logs/visual_dep_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="${TORCH_HOME:-$PWD/.torch_cache}"
export PYTHONPATH="${PWD}/src:${PWD}/companion/pascal:${PYTHONPATH:-}"

mkdir -p companion/pascal/logs companion/pascal/artifacts/visual_dep

PY="${HOME}/.conda/envs/vlm_doc_pascal_p1/bin/python"

# Mapping must already be frozen on CPU before this job.
if [[ ! -f companion/pascal/artifacts/visual_dep/unrelated_mapping.json ]]; then
  echo "ERROR: missing unrelated_mapping.json — freeze on CPU first" >&2
  exit 2
fi

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-none} CUDA=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L || true
"${PY}" -m visual_dep.runner
"${PY}" -m visual_dep.analyze
"${PY}" -m visual_dep.gallery
echo "VISUAL_DEP_SBATCH_DONE"
