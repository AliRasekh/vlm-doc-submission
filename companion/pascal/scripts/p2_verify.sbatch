#!/bin/bash
# Bounded P2.1 verification (strict CLI JSON, UI HTTP + browser). Shuts down before exit.
#SBATCH --job-name=vlm_p2_1_ver
#SBATCH --partition=volta
#SBATCH --account=pascal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=companion/pascal/logs/p2_1_verify_%j.out
#SBATCH --error=companion/pascal/logs/p2_1_verify_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export GRADIO_ANALYTICS_ENABLED=False
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"

PY="${HOME}/.conda/envs/vlm_doc_pascal_p1/bin/python"
mkdir -p companion/pascal/logs companion/pascal/artifacts/p2_1_verify

echo "host=$(hostname) job=${SLURM_JOB_ID:-na} rev=$(git rev-parse HEAD)"
nvidia-smi -L || true
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("cc", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
print("bf16_default", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)
print("bf16_no_emu", torch.cuda.is_bf16_supported(including_emulation=False) if torch.cuda.is_available() else None)
import gradio, gradio_client
print("gradio", gradio.__version__, "gradio_client", gradio_client.__version__)
try:
    import playwright
    print("playwright", playwright.__version__)
except Exception as exc:
    print("playwright_missing", exc)
PY

"$PY" companion/pascal/p2/verify_job.py
echo "P2.1 verification finished"
