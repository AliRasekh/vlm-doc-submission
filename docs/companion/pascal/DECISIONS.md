# Pascal companion decisions

## D1 — Isolation boundary

Tracked code/docs for Pascal experiments live only under `companion/pascal/` and `docs/companion/pascal/`. Shared `src/`, configs, tests, README, official results/report remain untouched.

## D2 — P1 is supplementary

P1 is exploratory development analysis designed after original holdout packaging. It does not alter the frozen deployment candidate (InternVL step 0) or official Phase 4 claims.

## D3 — Tile-cap candidates

Native configured maximum `M = max_num = 12` (`configs/models/internvl3_1b.yaml`).  
Predeclared unique sorted candidates: `{1, min(4, M), M}` → `{1, 4, 12}`.  
Do not raise M. Invalid caps vs `min_num=1` are omitted (none expected).

## D4 — Runtime on V100

Preserve Neumann settings where supported (`use_flash_attn=false`, greedy, `max_new_tokens=64`, same instruction).  
If bf16 unsupported, use documented **float16** with the same attention backend across all P1 conditions (no quantization, no CPU offload, no weight edits).

**Actual P1 job 117225:** default `torch.cuda.is_bf16_supported()` returned **True** on `Tesla V100-SXM3-32GB`, so **bfloat16** was used and float16 fallback was not applied. Attention remained eager.  
Post-hoc clarification (2026-10-03): PyTorch 2.5.1 defaults `including_emulation=True`; on CC 7.x V100 a default True indicates **BF16 usability/emulation**, not native CC≥8 hardware support. See `P1_RUNTIME_CLARIFICATION.md`.

## D5 — Measurement baseline

Paired quality deltas use the fresh Pascal original-cap (`max_num=12`) condition as control.  
Historical Neumann development predictions are compared for agreement diagnostics only; hardware/dtype differences are not treated as controlled latency comparisons.

## D6 — Execution order

Model loaded once. Conditions run sequentially on one GPU after identical warm-up each time, order: **12 → 4 → 1** (control first).

## D7 — P1 memory field caveat

P1 `infer_wrap` suppressed `reset_peak_memory_stats` during requests. Published tables use request-scoped peaks. Stored `gpu_memory_generation` must not be cited as generation-only under that wrapper.

## D8 — P2 demo runtime

P2 retains the executed P1 dtype (**bfloat16** + eager) for continuity, with accurate non-native labeling when CC<8 / `including_emulation=False` is False. No automatic precision switch from the default support boolean. No P1 memory monkey-patch. Loopback-only Gradio; no public share.

## D9 — P2.1 CLI JSON purity

`--json` must emit exactly one JSON document on stdout. Model/import/load notices go to stderr. Verification uses `json.loads` on the full stdout (no brace hunting, no raw-output fallback). Parse failure or CLI/engine prediction mismatch fails the job.

## D10 — P2.1 Gradio schema patch

Keep `gradio==5.9.1` / `gradio_client==1.5.2`. Patch `get_type` and `_json_schema_to_python_type` for bare bool schemas instead of blanking `get_api_info`. Do not upgrade the ML stack for UI schema issues.

## D11 — P2.1 SSH loopback forward

Document jump-host forwarding to `127.0.0.1:PORT` on the compute endpoint (`-L 127.0.0.1:PORT:127.0.0.1:PORT`). Do not use `-L PORT:COMPUTE_HOSTNAME:PORT` when the app binds loopback only. Do not widen the bind address.

## D12 — Pascal GitHub SSH key

Dedicated unencrypted Ed25519 key at `~/.ssh/id_ed25519_github_pascal_vlm_doc` for unattended repo-local Git (`IdentitiesOnly=yes`). No passphrase stored in the project. Global Git/SSH host config untouched.

## D13 — P3 three-seed fixed budget

Seeds **42 / 43 / 44** from corrected v2 (`seed: 42`). Report only update **200**. No hyperparameter/prompt/metric/tile tuning. No best-seed selection. No holdout. Neumann historical v2 u200 is a separate reference excluded from three-run mean/SD. Training outputs only under `companion/pascal/artifacts/p3/`.

## D14 — P3 BF16 policy

Retain original `prefer_bf16=true`. Do not auto-switch to FP16 because BF16 is non-native on V100. Proceed only if the existing gate passes on the selected GPU; otherwise stop and document a compatible-hardware alternative.

## D15 — P3 cross-node continuation

Job 117234 timeout forced seed-44 full restart on a later allocation (117252, pascal-node09 V100-PCIE) while seeds 42/43 remain from pascal-node10 V100-SXM3. Document as protocol deviation; do not claim single-GPU seed-only isolation across all three runs. Interrupted 164-update attempt preserved and excluded from completed stats.

## D16 — P3 outcome (development)

Under the frozen protocol, three Pascal u200 adapters show mean ANLS 0.8229 (sample SD 0.0076 across three seeds; **not** a confidence interval) vs fresh base 0.8276: two seeds below base, one slightly above. No consistent improvement; no best-seed selection; Neumann historical result kept separate.

## D17 — P3 analysis denominator

Primary P3 metrics use the development-manifest requested QID set as denominator. Failed and missing predictions count as zero. Duplicate or unexpected QIDs are rejected. Do not compute primary means on the base∩adapter intersection.

## D18 — Visual-input dependence (supplementary)

Post-holdout diagnostic on the original InternVL3-1B **base** only (not a P3 adapter): clean / white / unrelated images on all 208 development questions. Unrelated donors are frozen with seed 2026 before inference. Observed ~78 percentage-point ANLS drop under white/unrelated indicates strong dependence on correct visual input; not an exact decomposition, grounding proof, or contamination claim. Residual correct answers do not establish memorization. Distinct from P4 severity robustness.
