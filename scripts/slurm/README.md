# Slurm scripts (Neumann / ampere)

All GPU work must run here — not on the login node.

| Script | Job name | Purpose |
|--------|----------|---------|
| `smoke_neumann.sbatch` | `vlm_doc_smoke` | Phase 1 8-example smoke |
| `dev_smolvlm256m_neumann.sbatch` | `vlm_doc_dev256` | Phase 2A development baseline |
| `dev_smolvlm500m_neumann.sbatch` | `vlm_doc_dev500` | Phase 2B 500M development baseline |
| `dev_internvl3_1b_neumann.sbatch` | `vlm_doc_dev_ivl` | Phase 2C InternVL3-1B development baseline |
| `probe_internvl_generate_return.sbatch` | `vlm_doc_ivl_probe` | Phase 2D generate-return verification (2 qids) |

Common settings (observed authorized):

```text
--partition=ampere --account=ampere --qos=normal --gres=gpu:1 --cpus-per-task=8
```

Always exports `PYTHONNOUSERSITE=1` and uses `${VLM_DOC_ENV:-$HOME/.conda/envs/vlm_doc}/bin/python`.

```bash
mkdir -p logs results
sbatch scripts/slurm/dev_smolvlm256m_neumann.sbatch
```
