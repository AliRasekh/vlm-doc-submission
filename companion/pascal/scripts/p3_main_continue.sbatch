#!/bin/bash
# P3 continuation: resume incomplete seeds / evals (skip-existing in run_main).
#SBATCH --job-name=vlm_p3_cont
#SBATCH --partition=volta
#SBATCH --account=pascal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=companion/pascal/logs/p3_main_cont_%j.out
#SBATCH --error=companion/pascal/logs/p3_main_cont_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export PYTHONPATH="${PWD}/companion/pascal:${PYTHONPATH:-}"

PY="${HOME}/.conda/envs/vlm_doc_pascal_p1/bin/python"
mkdir -p companion/pascal/logs companion/pascal/artifacts/p3 companion/pascal/results

echo "host=$(hostname) job=${SLURM_JOB_ID:-na} rev=$(git rev-parse HEAD) CONTINUE"
nvidia-smi -L || true
test -f companion/pascal/artifacts/p3/gate_ok.json
"$PY" -m p3.run_main
echo "P3 continue finished"
