# Pascal companion plan

**Branch:** `companion/pascal-experiments`  
**Base (verified):** `0e94c20431355e52f047c29ebc3c98d8008a3747`  
**Ownership:** tracked work only under `companion/pascal/` and `docs/companion/pascal/`.

## Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| **P1** | Visual-budget ablation on frozen InternVL3-1B **base** (no LoRA), development set only | Completed (job 117225); runtime clarification 2026-10-03 |
| **P2** | Interactive demo using the verified backend (CLI + loopback UI) | Completed (job 117228) |
| **P3** | Fixed-budget three-seed corrected LoRA (dev eval only) | **Completed** (+ maintenance 2026-10-05) |
| **Visual-input dependence** | Base-model clean / white / unrelated-image diagnostic (dev 208) | **Completed** (job 117260) |
| **P4** | Severity-based robustness and structured error analysis | Not started |

## Non-goals

Holdout evaluation, merging into shared trees, publication, P4 execution from this workstream. P3 does **not** select a deployment seed.
