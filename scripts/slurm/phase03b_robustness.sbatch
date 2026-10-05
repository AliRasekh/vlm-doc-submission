#!/bin/bash
#SBATCH --job-name=vlm_doc_rob3b
#SBATCH --partition=ampere
#SBATCH --account=ampere
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vlm_doc_rob3b_%j.out
#SBATCH --error=logs/vlm_doc_rob3b_%j.err

# Phase 3B: six InternVL robustness evals (base+lora100 × clean/half/jpeg) on fixed subset.
# Does NOT run holdout. Does NOT train.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results/robustness_v1

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

if [ ! -f data/manifests/docvqa_dev_robustness_v1.json ]; then
  echo "ERROR: missing robustness subset manifest; run prepare_robustness_subset_v1.py first" >&2
  exit 2
fi
if [ ! -f results/phase03a_evidence_audit.json ]; then
  echo "ERROR: missing Phase 3A evidence audit" >&2
  exit 2
fi
"$PYTHON" - <<'PY'
import json
r=json.load(open("results/phase03a_evidence_audit.json"))
assert r.get("ok") is True, r
print("PHASE3A_EVIDENCE_OK")
PY

echo "=== ROBUSTNESS EVAL (6 combinations) ==="
"$PYTHON" scripts/run_robustness_dev_v1.py \
  --config configs/models/internvl3_1b.yaml \
  --manifest data/manifests/docvqa_dev_robustness_v1.json \
  --answers data/cache/docvqa_dev_v1_answers.json \
  --cache-dir "$HF_HOME" \
  --out-dir results/robustness_v1 \
  --models base,lora100

echo "=== ANALYZE ==="
"$PYTHON" scripts/analyze_robustness_v1.py \
  --out-dir results/robustness_v1 \
  --out-json results/robustness_v1_analysis.json \
  --out-md docs/examples/robustness_v1_gallery.md \
  --plot-svg docs/examples/robustness_v1_anls.svg \
  --example-dir docs/examples/robustness_v1_images

echo "PHASE3B_ROBUSTNESS_DONE"
