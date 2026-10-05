#!/bin/bash
# Pascal P1 visual-budget experiment (development only).
#SBATCH --job-name=vlm_p1_vb
#SBATCH --partition=volta
#SBATCH --account=pascal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=companion/pascal/logs/p1_%j.out
#SBATCH --error=companion/pascal/logs/p1_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="${TORCH_HOME:-$PWD/.torch_cache}"

PY="${VLAM_PY:-$HOME/.conda/envs/vlm_doc_pascal_p1/bin/python}"
mkdir -p companion/pascal/logs companion/pascal/artifacts/p1_runs companion/pascal/results

echo "host=$(hostname) job=${SLURM_JOB_ID:-na} cuda_visible=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L || true
"$PY" - <<'PY'
import os, torch
print("python", __import__('sys').version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("bf16_supported", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "mem", torch.cuda.get_device_properties(0).total_memory)
PY

echo "=== P1 smoke (limit 4, all caps) ==="
"$PY" companion/pascal/p1/runner.py --smoke \
  --out-dir companion/pascal/artifacts/p1_smoke

echo "=== P1 full development (208) ==="
"$PY" companion/pascal/p1/runner.py \
  --out-dir companion/pascal/artifacts/p1_runs

echo "=== P1 analyze ==="
"$PY" companion/pascal/p1/analyze.py \
  --run-dir companion/pascal/artifacts/p1_runs \
  --out-dir companion/pascal/results

echo "P1 job finished OK"
