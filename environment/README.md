# Environment recreation (Neumann / Python 3.11)

Isolated env path (local, not committed): `~/.conda/envs/vlm_doc`

**Always** set `PYTHONNOUSERSITE=1`. User-site packages previously polluted imports (wrong CUDA libs / torch builds).

```bash
micromamba create -y -p "$HOME/.conda/envs/vlm_doc" python=3.11 pip
export PYTHONNOUSERSITE=1
"$HOME/.conda/envs/vlm_doc/bin/pip" install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
"$HOME/.conda/envs/vlm_doc/bin/pip" install -r environment/requirements.txt
# torch pins sympy==1.13.1; if pip check complains, install that pin explicitly.
```

Verified stack: `environment/versions.verified.txt` and `environment/requirements.lock.txt`.

## Phase 2A environment checks (actual)

With `PYTHONNOUSERSITE=1` and the project interpreter:

- `pip check`: clean after pinning `sympy==1.13.1` (torch required 1.13.1; 1.14.0 had been present).
- Imports resolve under `~/.conda/envs/vlm_doc/lib/python3.11/site-packages/` (not `~/.local`).
- Slurm scripts set `PYTHONNOUSERSITE=1` and call `$HOME/.conda/envs/vlm_doc/bin/python`.

GPU inference is only valid inside Slurm ampere allocations.
