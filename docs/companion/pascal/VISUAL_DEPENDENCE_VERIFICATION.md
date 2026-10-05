# Visual-input dependence — verification

## Overlap audit

Existing Phase-3B robustness (`scripts/run_robustness_dev_v1.py`: clean / half_detail / jpeg_q40 on a robustness **subset**, optionally with LoRA) is **not** equivalent to this protocol (full 208, white geometry-matched blank, unrelated donor mapping seed 2026). **No silent reuse.**

## Scheduler

| Job | State | Elapsed | Node | Exit |
|-----|-------|---------|------|------|
| **117260** | COMPLETED | 00:25:18 | pascal-node09 | 0:0 |

Queue after completion: **empty**.

## Mapping

- Frozen before inference: `companion/pascal/artifacts/visual_dep/unrelated_mapping.json`
- `mapping_sha256=6d45e1b188f89e9a501213889b5381e417e0fa88d602184bd271d2fe34cce674`
- Eligibility: exclude same doc / same UCSF id / same image hash; prefer aspect-ratio match; no donor resize; answers unused.
- **Executed tie-break (deviation from original protocol wording):** among equal minimum aspect-distance donors, seeded `Random(2026).choice` is used (candidates ordered by QID). Original protocol text described ascending-QID-first then residual shuffle. Mapping **not** regenerated.

## Coverage

All three conditions: **208** unique QIDs, **0** duplicates, **0** missing, **0** failed.  
Generation-cap hits: clean 0; white 2; unrelated 1 — these examples remain in the requested-set ANLS/EM means.

White and clean tile-count histograms **matched in this run** (`{7:13, 10:3, 13:192}`); documented as an observation, not a protocol guarantee.

Prediction agreement with clean: denominator = pairs where **both** records are `status=ok` (not the full 208 if some fail/miss). Agreement ≠ reference correctness (ANLS/EM).

## Protocol / source

| Item | SHA |
|------|-----|
| Protocol text committed pre-inference | in `5a4c517…` (`VISUAL_DEPENDENCE_PROTOCOL.md`) |
| Job source revision (runtime.json) | `5a4c5174feb5331ed641480cd756b81d8654d096` |
| Scientific settings vs verified base eval | matched (instruction, greedy, max_new_tokens=64, max_num=12, thumbnail, model revision) |

## Deviations

1. **Mapping tie-break wording vs code** (see Mapping above): original protocol described ascending donor QID then residual seeded shuffle; executed code uses seeded choice among equal-aspect-distance donors. Frozen mapping retained.
2. Physical GPU is V100-PCIE (node09), consistent with recent P3 evals; not a multi-node split within this experiment.

## Non-claims recorded

“Replacing the correct page with a white or unrelated image reduced development ANLS by approximately 78 percentage points, indicating strong dependence on the correct visual input.”  
Not an exact performance decomposition, fine-grained grounding proof, or contamination/memorization claim. Residual corrects ≠ memorization proof; unrelated ≠ pure content ablation; not P4.
