# Pascal companion experiments

Isolated ownership surface for Pascal-side supplementary work.  
Do **not** modify shared `src/`, root configs, official results, or report trees from here.

## Roadmap

See [`docs/companion/pascal/PLAN.md`](../../docs/companion/pascal/PLAN.md).

| Phase | Topic | Status |
|-------|-------|--------|
| P1 | Visual-budget (`max_num`) ablation — InternVL3-1B base, development only | **Completed** (+ runtime clarification) |
| P2 | Interactive demo (CLI + Gradio UI) | **Completed** (job 117228); **P2.1** defect close-out |
| P3 | Fixed-budget three-seed LoRA reproducibility (dev only) | **Completed** (+ maintenance) |
| Visual-input dependence | Base clean / white / unrelated images (dev only) | **Completed** (job 117260) |
| P4 | Severity robustness / structured errors (later) | Not started |

## Environment

ML env (P1): `$HOME/.conda/envs/vlm_doc_pascal_p1`

```bash
export PYTHONNOUSERSITE=1
# ML pins from repo environment/ (see P1 README section historically)
"$HOME/.conda/envs/vlm_doc_pascal_p1/bin/pip" install -r companion/pascal/requirements-ui.txt
# Keep ML Pillow pin (Gradio 4.44.1 may warn about pillow<11):
"$HOME/.conda/envs/vlm_doc_pascal_p1/bin/pip" install 'Pillow==11.1.0'
```

Always set `PYTHONNOUSERSITE=1`. GPU inference only via Slurm.

## P1

- Protocol: `docs/companion/pascal/P1_PROTOCOL.md` (immutable)
- Clarification: `docs/companion/pascal/P1_RUNTIME_CLARIFICATION.md`
- Results: `docs/companion/pascal/P1_RESULTS.md`
- Compact table/figures: `companion/pascal/results/`

## P2 demo

Docs: `docs/companion/pascal/P2_DEMO.md`, verification: `docs/companion/pascal/P2_VERIFICATION.md`.

```bash
export PYTHONNOUSERSITE=1
# CPU tests
"$HOME/.conda/envs/vlm_doc_pascal_p1/bin/python" -m pytest companion/pascal/tests -q

# CLI (inside a GPU allocation)
"$HOME/.conda/envs/vlm_doc_pascal_p1/bin/python" companion/pascal/p2/cli.py \
  --preset exact_success --max-num 12 --json

# Bounded P2.1 verify (strict CLI JSON, UI HTTP + browser, shuts down)
sbatch companion/pascal/scripts/p2_verify.sbatch

# Interactive demo session (loopback on compute node)
sbatch companion/pascal/scripts/p2_demo.sbatch
```

SSH forward — server binds **127.0.0.1** on the compute node. From the laptop, use a jump host and forward to **loopback on the final compute endpoint** (not `COMPUTE_HOSTNAME:7860` as the `-L` target). Username on-site: `ali.rasekh`. Login host is typically `pascal.l3s.intra` when already on the cluster network; replace `LOGIN_HOST` if your entry node differs. Compute hostname comes from the job log.

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -J ali.rasekh@LOGIN_HOST \
  -L 127.0.0.1:7860:127.0.0.1:7860 \
  ali.rasekh@COMPUTE_HOSTNAME
# open http://127.0.0.1:7860/
```

Laptop→cluster tunnel end-to-end from an external laptop was **not** exercised in P2.1; instructions and launcher text were corrected.

## P3 training seeds

Protocol: `docs/companion/pascal/P3_PROTOCOL.md`. Seeds **42 / 43 / 44** (original corrected v2 seed and +1/+2).

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/companion/pascal:${PYTHONPATH:-}"
"$HOME/.conda/envs/vlm_doc_pascal_p1/bin/python" -m p3.freeze_targets
sbatch companion/pascal/scripts/p3_preflight.sbatch   # gate + timing probe
# after gate_ok:
sbatch companion/pascal/scripts/p3_main.sbatch         # three trains + four-system dev eval
```

Artifacts (gitignored): `companion/pascal/artifacts/p3/`. Compact tables/figures: `companion/pascal/results/`.  
Results: `docs/companion/pascal/P3_RESULTS.md`, verification: `docs/companion/pascal/P3_VERIFICATION.md`.

## Visual-input dependence

Protocol: `docs/companion/pascal/VISUAL_DEPENDENCE_PROTOCOL.md` (base InternVL3-1B only).

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PWD/companion/pascal:${PYTHONPATH:-}"
"$HOME/.conda/envs/vlm_doc_pascal_p1/bin/python" -m visual_dep.freeze_mapping
sbatch companion/pascal/scripts/visual_dependence.sbatch
```

## Labeling

P1/P2/P3 and visual-input dependence are **supplementary**. Original holdout results and model selection remain unchanged.
