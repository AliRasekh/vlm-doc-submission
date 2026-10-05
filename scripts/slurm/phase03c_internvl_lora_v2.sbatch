#!/bin/bash
#SBATCH --job-name=vlm_doc_lora3c
#SBATCH --partition=ampere
#SBATCH --account=ampere
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/vlm_doc_lora3c_%j.out
#SBATCH --error=logs/vlm_doc_lora3c_%j.err

# Phase 3C: corrected termination LoRA v2 — gate + train + dev eval + compare + robustness.
# Does NOT overwrite v1 checkpoints/logs/predictions. Does NOT run holdout.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results checkpoints/internvl3_1b_lora_v2 results/robustness_v2

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
unset TRANSFORMERS_CACHE || true
export TORCH_HOME="${TORCH_HOME:-$PWD/.torch_cache}"
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME"

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
PY

for p in \
  results/train_internvl3_1b_lora_v1.jsonl \
  results/train_lora_v1_supervised_example.json \
  checkpoints/internvl3_1b_lora_v1/update_0100/adapter_model.safetensors \
  results/dev_internvl_lora_v1_comparison.json \
  data/manifests/docvqa_dev_robustness_v1.json
do
  if [ ! -f "$p" ]; then
    echo "ERROR: missing preserved artifact: $p" >&2
    exit 2
  fi
done

"$PYTHON" - <<'PY'
from pathlib import Path
import json
p=Path("results/train_internvl3_1b_lora_v1.jsonl")
meta={"path":str(p),"nbytes":p.stat().st_size,"mtime":p.stat().st_mtime}
Path("results/phase03c_v1_log_guard.json").write_text(json.dumps(meta,indent=2)+"\n")
print("V1_LOG_GUARD", meta)
PY

echo "=== UNIT TESTS (termination) ==="
"$PYTHON" -m unittest tests.test_collator_termination -v

echo "=== CORRECTNESS GATE (v2) ==="
"$PYTHON" scripts/gate_internvl_lora.py \
  --config configs/train/internvl3_1b_lora_v2.yaml \
  --cache-dir "$HF_HOME" \
  --out results/internvl_lora_v2_gate_ok.json

echo "=== MAIN TRAIN 200 UPDATES (v2) ==="
"$PYTHON" scripts/train_internvl_lora.py \
  --config configs/train/internvl3_1b_lora_v2.yaml \
  --cache-dir "$HF_HOME"

echo "=== DEV EVAL STEP 100 (v2) ==="
"$PYTHON" scripts/run_eval.py \
  --config configs/models/internvl3_1b.yaml \
  --manifest data/manifests/docvqa_dev_v1.json \
  --answers data/cache/docvqa_dev_v1_answers.json \
  --out-jsonl results/dev_internvl3_1b_lora_v2_u100.jsonl \
  --summary-json results/dev_internvl3_1b_lora_v2_u100_summary.json \
  --cache-dir "$HF_HOME" \
  --lora-adapter checkpoints/internvl3_1b_lora_v2/update_0100 \
  --train-update 100 \
  --score

echo "=== DEV EVAL STEP 200 (v2) ==="
"$PYTHON" scripts/run_eval.py \
  --config configs/models/internvl3_1b.yaml \
  --manifest data/manifests/docvqa_dev_v1.json \
  --answers data/cache/docvqa_dev_v1_answers.json \
  --out-jsonl results/dev_internvl3_1b_lora_v2_u200.jsonl \
  --summary-json results/dev_internvl3_1b_lora_v2_u200_summary.json \
  --cache-dir "$HF_HOME" \
  --lora-adapter checkpoints/internvl3_1b_lora_v2/update_0200 \
  --train-update 200 \
  --score

echo "=== COMPARE v2 vs step0 ==="
"$PYTHON" scripts/compare_internvl_lora_dev.py \
  --baseline results/dev_internvl3_1b.jsonl \
  --step100 results/dev_internvl3_1b_lora_v2_u100.jsonl \
  --step200 results/dev_internvl3_1b_lora_v2_u200.jsonl \
  --out results/dev_internvl_lora_v2_comparison.json \
  --examples-md docs/examples/lora_v2_before_after.md

"$PYTHON" - <<'PY'
import json
from pathlib import Path
g=json.load(open("results/phase03c_v1_log_guard.json"))
p=Path(g["path"])
assert p.stat().st_size==g["nbytes"] and abs(p.stat().st_mtime-g["mtime"])<1e-6
print("V1_LOG_PRESERVED_OK")
cmp=json.load(open("results/dev_internvl_lora_v2_comparison.json"))
best=int(cmp["best_trained_step"])
Path("results/phase03c_best_trained_step.txt").write_text(str(best)+"\n")
print("BEST_TRAINED_STEP", best)
print("SELECTED_DEV_STEP", cmp["selected_step_for_dev"], "ft_improved", cmp.get("fine_tuning_improved_selection"))
PY

BEST=$(tr -d '[:space:]' < results/phase03c_best_trained_step.txt)
echo "=== ROBUSTNESS best trained v2 update ${BEST} ==="
"$PYTHON" scripts/run_robustness_dev_v1.py \
  --config configs/models/internvl3_1b.yaml \
  --manifest data/manifests/docvqa_dev_robustness_v1.json \
  --answers data/cache/docvqa_dev_v1_answers.json \
  --cache-dir "$HF_HOME" \
  --out-dir results/robustness_v2 \
  --lora-checkpoint-root checkpoints/internvl3_1b_lora_v2 \
  --models "lora${BEST}"

echo "PHASE3C_PIPELINE_DONE"
