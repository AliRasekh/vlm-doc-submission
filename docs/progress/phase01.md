# Phase 1 progress — Git, environment, DocVQA smoke

## Goal

Establish a reviewable repository, an isolated PyTorch environment, pinned SmolVLM-256M assets, an 8-example DocVQA smoke set, and one ampere Slurm smoke inference job.

## Timeline

### Prior Phase 0 (not re-executed)

- **When:** 2026-09-30 ~08:00+02, re-verified ~08:11+02 (from local audit timestamps).
- **Evidence:** untracked `phase0_neumann.local.md`; probe job **81806** COMPLETED (`ampere` / `ampere` / `normal`, A100 80GB, torch 2.5.1 / CUDA 11.8, BF16 OK).
- Hopper/blackwell account combinations rejected; standby not used in Phase 1.

### Phase 1 execution

#### Git

- Initialized empty local repository on `main`.
- Remote set to `git@github.com:AliRasekh/vlm-doc.git`.
- Remote refs via SSH with default and AliRasekh-named identities: **Permission denied (publickey)**.
- One other local identity authenticates as a different GitHub account and cannot see the repo (`ERROR: Repository not found`).
- Public HTTPS/API to `AliRasekh/vlm-doc` returns **404** (private, empty, or nonexistent from this network’s unauthenticated view). **Visibility: unverified.**
- No keys created; no auth settings changed; no force-push.

#### Environment

- Created isolated env at `~/.conda/envs/vlm_doc` (Python 3.11) via micromamba; did **not** modify `gpu` / `qwen`.
- Required `PYTHONNOUSERSITE=1` because user-site packages otherwise polluted imports.
- Accidental upgrade to torch 2.14+cu130 during an unpinned reinstall was **reverted**; pinned stack restored to **torch 2.5.1+cu118**.
- Verified versions: `environment/versions.verified.txt`.

#### Model download

- Model: `HuggingFaceTB/SmolVLM-256M-Instruct`
- Immutable revision: `7e3e67edbbed1bf9888184d9df282b700a323964`
- Hub client successfully downloaded metadata and `model.safetensors` (~513 028 808 bytes) into project `.hf_cache/` (persistent, gitignored).
- Phase 0 guessed CDN hostname DNS failures did **not** block the official Hub client download on this attempt.

#### Dataset / smoke manifest

- VLMEvalKit commit inspected: `54a063c5bf75245c35599daab9826d0a5b8e2671` (`DocVQA_VAL` → OpenCompass TSV).
- OpenCompass HTTPS: **SSL certificate problem: certificate has expired** (TLS not disabled).
- Alternative: `HuggingFaceM4/DocumentVQA` revision `a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4`, single shard `data/validation-00000-of-00017.parquet` (~46 MB), not train.
- Selection: sort by `questionId`; indices `round(i*(n-1)/7)` for `i=0..7` on `n=315` → `[0,45,90,135,179,224,269,314]`.
- Example `question_id`s: `223, 383, 16606, 32897, 49394, 57368, 57529, 58702`.
- Images + answer sidecar gitignored; tracked manifest: `data/manifests/docvqa_smoke_v1.json`.

#### Smoke job

- Script: `scripts/slurm/smoke_neumann.sbatch` (`vlm_doc_smoke`, ampere/ampere/normal, 1 GPU, 30 min).
- **Job ID:** 81832
- **State:** COMPLETED ExitCode 0:0, Elapsed 00:00:42, Node `gpunode05`, MaxRSS ~2319832K
- **GPU:** NVIDIA A100 80GB PCIe (81920 MiB); driver 615.71.09; BF16 supported and used
- **Parameters:** total unique **256484928**
- **Code revision at run:** `c332fad03b3adcc29655462568d606d5bfe6d4d0` (clean tree)
- **Results:** 8/8 `status=ok` in local `results/smoke_smolvlm256m.jsonl` (gitignored)
- Smoke Q/ref/pred pairs recorded in README handoff / this phase note (diagnostics only, **not** a benchmark)

| question_id | question (short) | reference (first) | prediction |
|-------------|------------------|-------------------|------------|
| 223 | title of the chart? | SARGASSO SEA TEMPERATURE | Unsettled Science. |
| 383 | which nitrosamine… tobacco? | NNK | Nitrosamine |
| 16606 | type of form? | PROJECT ASSIGNMENT FORM | Project Assignment Form |
| 32897 | memorandum addressed to? | Volunteers Against Hunger Steering Committee | Volunteers Against Hunger Steering Committee. |
| 49394 | full form of IUNS? | international union of nutritional sciences | International Union of Nutritional Sciences |
| 57368 | nomination meetings attended? | 2 | 2 |
| 57529 | ITC Brand Liquid Crystal…? | Fiama Di Wills | ITC Limited |
| 58702 | % net pounds out/infeed? | 83.4% | 83.4 %. |

## Artifacts (tracked vs local)

| Path | Tracked? |
|------|----------|
| `src/`, `scripts/`, `configs/`, `environment/`, `docs/`, `README.md`, `.gitignore` | yes |
| `data/manifests/docvqa_smoke_v1.json` | yes |
| `.hf_cache/`, `data/images/`, `data/cache/`, `results/*.jsonl`, `logs/` | no |
| `phase0_neumann.local.md`, probe leftovers | no |
