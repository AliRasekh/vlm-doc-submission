#!/bin/bash
#SBATCH --job-name=vlm_doc_img_audit
#SBATCH --partition=ampere
#SBATCH --account=ampere
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/vlm_doc_img_audit_%j.out
#SBATCH --error=logs/vlm_doc_img_audit_%j.err
# CPU-only (verified ampere accepts no-GPU jobs; probe 82978)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
ENV_PREFIX="${VLM_DOC_ENV:-$HOME/.conda/envs/vlm_doc}"
PYTHON="${ENV_PREFIX}/bin/python"
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} job=${SLURM_JOB_ID:-}"
"$PYTHON" scripts/audit_train_subset_images_v1.py \
  --manifest data/manifests/docvqa_train_subset_v1.json \
  --out results/train_subset_v1_image_integrity_audit.json
echo "IMAGE_AUDIT_DONE"
