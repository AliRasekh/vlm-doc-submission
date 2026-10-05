#!/bin/bash
# P3 compatibility gate + short timing probe (discards probe adapters).
#SBATCH --job-name=vlm_p3_pre
#SBATCH --partition=volta
#SBATCH --account=pascal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=companion/pascal/logs/p3_preflight_%j.out
#SBATCH --error=companion/pascal/logs/p3_preflight_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"

PY="${HOME}/.conda/envs/vlm_doc_pascal_p1/bin/python"
export PYTHONPATH="${PWD}/companion/pascal:${PYTHONPATH:-}"
mkdir -p companion/pascal/logs companion/pascal/artifacts/p3

echo "host=$(hostname) job=${SLURM_JOB_ID:-na} rev=$(git rev-parse HEAD)"
nvidia-smi -L || true
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("cc", torch.cuda.get_device_capability(0))
print("bf16_default", torch.cuda.is_bf16_supported())
print("bf16_no_emu", torch.cuda.is_bf16_supported(including_emulation=False))
PY

"$PY" -m p3.preflight
echo "P3 preflight finished"
