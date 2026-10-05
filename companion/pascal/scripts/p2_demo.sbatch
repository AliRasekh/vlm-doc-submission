#!/bin/bash
# Interactive P2 demo launcher (user session). Binds loopback on the compute node.
#SBATCH --job-name=vlm_p2_demo
#SBATCH --partition=volta
#SBATCH --account=pascal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=companion/pascal/logs/p2_demo_%j.out
#SBATCH --error=companion/pascal/logs/p2_demo_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export GRADIO_ANALYTICS_ENABLED=False

PY="${HOME}/.conda/envs/vlm_doc_pascal_p1/bin/python"
PORT="${P2_PORT:-7860}"
HOST="127.0.0.1"
USER_NAME="${USER:-ali.rasekh}"
# Cluster login hostname as seen on-site (not an externally invented FQDN).
LOGIN_HINT="pascal.l3s.intra"

echo "============================================================"
echo "P2 demo on $(hostname) job=${SLURM_JOB_ID:-na}"
echo "UI: http://${HOST}:${PORT}/  (loopback on compute node only)"
echo ""
echo "SSH forward from your laptop (jump host → compute loopback)."
echo "Replace LOGIN_HOST if your entry node differs from ${LOGIN_HINT}."
echo "Forward destination is evaluated on the final compute SSH endpoint:"
echo ""
echo "  ssh -N -o ExitOnForwardFailure=yes \\"
echo "    -J ${USER_NAME}@LOGIN_HOST \\"
echo "    -L 127.0.0.1:${PORT}:127.0.0.1:${PORT} \\"
echo "    ${USER_NAME}@${HOSTNAME}"
echo ""
echo "Then open http://127.0.0.1:${PORT}/ on the laptop."
echo ""
echo "Two-stage alternative (distinct intermediate port 17860 on login):"
echo "  # on laptop:"
echo "  ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:${PORT}:127.0.0.1:17860 ${USER_NAME}@LOGIN_HOST"
echo "  # on login node:"
echo "  ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:17860:127.0.0.1:${PORT} ${USER_NAME}@${HOSTNAME}"
echo "============================================================"

nvidia-smi -L || true
exec "$PY" companion/pascal/p2/app.py --host "$HOST" --port "$PORT" --preload
