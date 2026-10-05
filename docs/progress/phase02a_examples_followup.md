# Phase 2A documentation follow-up — examples gallery & metric/latency clarification

**When:** 2026-09-30 ~09:50+02  
**Scope:** Documentation and scoring clarification only. No new inference, no holdout access, manifests unchanged.

## Actions

1. Added `docs/examples/README.md` with six reproducible smoke/dev examples and a CPU-reconstructed transform walkthrough for qid `5158`.
2. Clarified primary ANLS cutoff: applied to **similarity** via `sim >= 0.5` (equivalently keep when **NL ≤ 0.5**). At **NL == 0.5**, score is **0.5**. Corrected an earlier docstring that falsely equated this with pix2struct’s `NL < 0.5` (which would zero NL == 0.5). **Implementation unchanged; no rescore required; development aggregates not superseded.**
3. Documented latency as **generation-only** (`model.generate` with CUDA sync), excluding preprocess/H2D/decode; warm-up excluded from aggregates.
4. Added ongoing examples documentation rule (fine-tune / before-after / robustness marked planned).
5. Regenerated local `results/phase02a_review.txt`.

## Selected gallery IDs

`5158`, `292`, `297`, `57368`, `5263`, `53842`
