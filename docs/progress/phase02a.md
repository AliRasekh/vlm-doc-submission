# Phase 2A progress — audit, splits, metrics, SmolVLM-256M development baseline

## Goal

Stabilize the Phase 1 smoke pipeline, freeze DocVQA development (~200) and holdout (~500) manifests, implement ANLS/EM with tests, and run SmolVLM-256M on the development set only.

## Timeline

### Git / SSH (2026-09-30 ~09:12–09:14+02)

- Created dedicated Ed25519 key `~/.ssh/id_ed25519_vlm_doc_neumann` (passphrase empty; mode 600).
- Repo-local `core.sshCommand` with `IdentitiesOnly=yes` (no global SSH config changes).
- User confirmed GitHub Authentication Key install.
- Verified: `Hi AliRasekh! You've successfully authenticated`.
- Remote was empty; pushed `main` → remote commit **`b5d2ec498d125c66d868dc0e1b418f55a4f8be35`** matches local HEAD at push time.
- Unauthenticated API still returns 404 → treat visibility as **private/unverified** via public API.

### Audit findings and fixes

| Finding | Severity | Fix |
|---------|----------|-----|
| Resume skipped any `status=ok` by `question_id` only — could reuse predictions across model/manifest/prompt/gen changes | High | `run_fingerprint` must match (model id/revision, manifest sha, instruction, max_new_tokens, do_sample, prefer_bf16, seed, config path) |
| Failed examples written, but aggregate path incomplete for evaluation | Medium | `aggregate_scores(..., n_requested=)` counts failures/missing as zeros |
| Docs referred to MaxRSS near GPU discussion without clarifying host vs GPU | Doc | Explicit: MaxRSS = host RSS; add PyTorch peak allocated/reserved with scope notes |
| Smoke summary omitted GPU allocator peaks | Medium | Record per-example `gpu_memory` via `torch.cuda.max_memory_*` after `reset_peak_memory_stats` around `generate()` |
| `PYTHONNOUSERSITE` set in Slurm but not in Python entrypoints | Low | `run_eval.py` forces `PYTHONNOUSERSITE=1` |
| `pip check`: sympy 1.14 vs torch pin 1.13.1 | Low | Targeted reinstall `sympy==1.13.1`; `pip check` clean afterward |
| Prior smoke JSONL has no fingerprints | Implication | Phase 1 smoke outputs remain valid as diagnostics but are **not** safely resumable under the new rules; re-runs need a new out path or matching fingerprint |

Verified still true: references never enter model inputs; only new tokens decoded; `model.eval()` + `inference_mode()`; greedy `do_sample=False`; revisions pinned in YAML; Slurm uses `$HOME/.conda/envs/vlm_doc` with `PYTHONNOUSERSITE=1`.

### Environment check (login, `PYTHONNOUSERSITE=1`)

- Imports from env site-packages (not `~/.local`): torch 2.5.1+cu118, torchvision 0.20.1+cu118, transformers 4.48.3.
- `pip check`: no broken requirements after sympy pin.

### Metrics

- Primary ANLS: lowercase + strip + collapse whitespace; NL = edit / max(len); score = 1−NL if ≥ 0.5 else 0; max over refs; empty vs nonempty → 0.
- Compatibility: `anls_vlmevalkit` (VLMEvalKit commit `54a063c5…` denominator quirk).
- EM: same normalization; equality; no punctuation stripping.
- Tests: `tests/test_metrics.py` — 19 tests OK.

### Data splits

- Full validation (17 shards, 5349 questions) from `HuggingFaceM4/DocumentVQA` @ `a44195f8…`.
- Seed **42**; group by **`doc_id`**; smoke docs excluded from both; holdout ⊥ development by `doc_id`.
- Development: **208** questions / **48** docs — manifest sha256 `7302844ae058d7fa3566c418d3770b90fbd3433fb555806f9231de1fc576aab8`.
- Holdout: **501** questions / **112** docs — manifest sha256 `e9f03fe23dcdccb97e17871e37a6e621ca7beaebe372c6decb1e51bd1bf714d9`.
- Image SHA256 overlap across splits: **0**.
- Train: 38 shards listed; **not downloaded**.

### Development baseline job

- Script: `scripts/slurm/dev_smolvlm256m_neumann.sbatch`
- **Job ID:** 81847
- **State:** COMPLETED ExitCode 0:0, Elapsed 00:02:59, Node `gpunode03`
- **GPU:** NVIDIA A100-PCIE-40GB; BF16; params 256484928
- **Host MaxRSS:** 2268708K (~2.27 GiB) — **host RAM**, not GPU
- **Peak GPU allocated/reserved:** 2984801792 / 3550478336 bytes (PyTorch allocator)
- **Code revision:** `9da8a8f2eddd47667ab796659f65b87abdf107f3` (clean)
- **Scores (requested means):** ANLS **0.5664**, EM **0.2596**, VLMEvalKit-compat ANLS 0.5665; 208/208 ok
- **Latency:** median 0.209 s, p95 0.334 s (warmup excluded)
- Local artifacts: `results/dev_smolvlm256m.jsonl`, `results/dev_smolvlm256m_summary.json`, `results/phase02a_review.txt` (gitignored)

## Artifacts

| Path | Tracked? |
|------|----------|
| manifests, metrics, eval runner, docs | yes |
| images, answer sidecars, JSONL, review packet, HF cache | no |