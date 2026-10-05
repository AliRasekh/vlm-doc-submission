# Pascal companion progress

Chronological log for `companion/pascal-experiments`.

## 2026-10-03 — P1 completed

- Experiment revision `cacbb90…`; results commit `7f703ec…`; job **117225** COMPLETED.
- Caps `{1,4,12}` on InternVL3-1B base, development 208.

## 2026-10-03 — P1 runtime clarification + P2 start

- Inspected PyTorch 2.5.1: `is_bf16_supported(including_emulation=True)` **by default**; V100 default True is usability/emulation, not native CC≥8.
- Documented P1 memory monkey-patch caveat: published peaks are request-scoped; `gpu_memory_generation` not independently generation-only under the wrapper.
- Wrote `P1_RUNTIME_CLARIFICATION.md`; corrected present-tense misleading claims.
- Implemented P2 CLI + Gradio loopback UI under `companion/pascal/p2/` (no memory monkey-patch).
- CPU: 9 pytest tests passed; Gradio Blocks builds; UI deps installed with Pillow restored to 11.1.0.

## 2026-10-03 — P2 GPU verification

- Job **117228** COMPLETED on pascal-node10 / V100-SXM3-32GB, elapsed 00:00:56.
- BF16: default True, `including_emulation=False` **False**, CC 7.0 — usability/emulation, not native.
- CLI+engine cap12 answer match; caps 12→4 tile 13→5; serialization OK; UI HTTP OK; server shut down.
- Browser visual QA not performed (HTTP only).
- Later found: CLI stdout polluted by FlashAttention notice; verify used `rfind("{")`; Gradio `get_api_info` stubbed; SSH `-L` docs wrong for loopback bind.

## 2026-10-03 — P2.1 defect close-out

- Dedicated GitHub Ed25519 key `~/.ssh/id_ed25519_github_pascal_vlm_doc` (comment `pascal-vlm-doc`, unencrypted for unattended Git); repo-local `core.sshCommand` with `IdentitiesOnly`; prior unset/default→stale Neumann identity recorded in `companion/pascal/ssh_github_local.md`.
- Strict CLI JSON (`cli_json.py`); notices → stderr; verify fails on parse/mismatch.
- Gradio schema patch restores `get_api_info` (bool `additionalProperties`); references by image SHA256+question.
- SSH forward docs/launcher: jump + `127.0.0.1:7860:127.0.0.1:7860` on compute endpoint.
- Job **117232** COMPLETED on pascal-node10 / V100 (00:01:30): strict CLI match; browser+Client UI QA passed; 6 screenshots under `results/p2_1_screenshots/`; server shut down; queue clear.

## 2026-10-03 — P2.1 push + P3 start

- GitHub SSH auth succeeded; pushed `companion/pascal-experiments` @ `845b7bf…` (remote SHA matched).
- P3 protocol frozen: seeds **42, 43, 44**; thin wrappers under `companion/pascal/p3/`; no holdout; original v2 settings retained (`prefer_bf16=true`).

## 2026-10-04/05 — P3 execution and close-out

- Gate **117233** PASSED on node10 V100-SXM3 (BF16 usability; non-native).
- Main **117234** TIMEOUT: seeds 42/43 u200 complete; seed 44 interrupted at 164 (preserved under `artifacts/p3/interrupted/`).
- Continue **117252** COMPLETED on node09 V100-PCIE: seed 44 full restart + four-system dev eval + analysis.
- Dev ANLS: base 0.8276; seeds 0.8193 / 0.8178 / 0.8317; mean 0.8229 ± 0.0076 (**sample SD**, n=3 seeds; not a CI). No consistent gain; no seed selection.
- Protocol deviation: different physical nodes/GPU SKUs (SXM3 vs PCIE) across seeds — documented.

## 2026-10-05 — P3 maintenance close-out

- Coverage wording: 1600 unique QIDs = **80%** of the 2000-pool (not full coverage).
- `analyze.py`: requested-manifest denominator; failed/missing→0; reject dup/unexpected QIDs; no intersection primary metrics; fresh-base counts from `scores.n_completed_ok` etc.
- `run_main.py`: skip-existing validates coverage + fingerprints; incompatible caches preserved (no overwrite/auto-rerun).
- Figures: raw loss faint + trailing 20-update mean; ANLS label margin for seed 44.
- Recomputed from existing predictions: published ANLS/EM **unchanged**.

## 2026-10-05 — Visual-input dependence (start)

- Protocol frozen under `VISUAL_DEPENDENCE_PROTOCOL.md` (base model only; clean/white/unrelated).
- Prior Phase-3B robustness (`half_detail`/`jpeg_q40` subset) is **not** equivalent — no silent reuse.
- Distinct from planned P4 severity-robustness (still not started).

## 2026-10-05 — Visual-input dependence (completed)

- Job **117260** COMPLETED on pascal-node09 / V100-PCIE (00:25:18); source `5a4c517…`.
- Dev ANLS: clean 0.8276; white 0.0421 (Δ −0.7855); unrelated 0.0469 (Δ −0.7807).
- Summary: replacing the correct page with a white or unrelated image reduced development ANLS by ~78 percentage points → strong dependence on correct visual input (not an exact decomposition, grounding proof, or contamination claim).
## 2026-10-05 — Review close-out (CPU)

- `visual_dep/analyze.py`: manifest requested QIDs; reject dup/unexpected; agreement only on both-ok pairs; assert vs condition summaries.
- P3 `validate_eval_cache`: required provenance present; record fingerprint coherence; mismatched/missing fingerprints → incompatible (preserve, no overwrite).
- Mapping tie-break protocol clarification/deviation documented; mapping not regenerated.
- CPU regression tests refreshed into review packets.

