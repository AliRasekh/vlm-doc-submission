# Phase 2B CPU follow-up — EM drop analysis & gallery provenance

**Date:** 2026-09-30  
**Scope:** CPU-only analysis of existing development predictions. Primary metrics, predictions, prompts, development membership, and holdout unchanged. No new inference or training.

Script: `scripts/analyze_phase02b_cpu_followup.py`  
Artifact: `results/phase02b_cpu_followup_analysis.json`

## 1. Paired comparison validation

Joined 256M and 500M JSONL predictions by `question_id` against `docvqa_dev_v1` and `data/cache/docvqa_dev_v1_answers.json`.

| Check | Result |
|-------|--------|
| Unique requested IDs per model | **208 / 208** |
| Questions match manifest | yes |
| References match sidecar | yes |
| Recomputed strict-v2 ANLS means | 256M **0.5568**, 500M **0.6095** (match prior) |
| Recomputed primary EM | 256M **0.2596** (54/208), 500M **0.0433** (9/208) |
| Improved / regressed / tied | **53 / 70 / 85** (match prior) |

### Why mean ANLS rises despite more regressions

| Quantity | Value |
|----------|------:|
| Σ positive per-question ΔANLS (500−256) | **+33.885** |
| Σ negative per-question ΔANLS | **−22.920** |
| Net mean Δ (= (Σpos+Σneg)/208) | **+0.0527** |

Larger improvements outweigh a greater number of smaller regressions.

## 2. Formatting diagnostic (not a primary metric)

After existing lowercase/strip/whitespace-collapse normalization, remove **at most one** trailing ASCII `.` from both prediction and each reference. No removal of internal punctuation, decimals, `%`, units, articles, or arbitrary suffixes.

| Model | Primary EM | Diagnostic EM | Newly matched | Still unmatched | Preds ending in `.` |
|-------|------------|---------------|---------------|-----------------|---------------------|
| 256M | 54 / 0.2596 | 103 / 0.4952 | 49 | 105 | 98 |
| 500M | 9 / 0.0433 | 131 / 0.6298 | 122 | 77 | 197 |

**Limitation:** terminal punctuation can be meaningful; this diagnostic is deliberately limited and is **not** official EM or official benchmark accuracy. Primary and diagnostic scores are retained separately.

**Interpretation:** The diagnostic shows **substantial sensitivity to answer formatting**. Under this limited terminal-period rule, ranking **reverses** (500M 131/208 vs 256M 103/208) relative to primary EM (500M 9/208 vs 256M 54/208). That reversal itself demonstrates strong formatting sensitivity. It does **not** establish that every mismatch is punctuation-related, nor that diagnostic EM is official accuracy. Residual content errors remain under the diagnostic (e.g. qid `5263` `Mr. tom rudley.`).

Deterministic samples (ascending qid): see analysis JSON `formatting_diagnostic.*.sample_*`.

## 3. Output behavior

| | 256M | 500M |
|--|------|------|
| Gen-token length available | 208 | 208 |
| min / median / p95 / max | 2 / 6 / ~13.6 / 56 | 3 / 6 / 14 / 22 |
| `generation_cap_reached` | **unavailable** on Phase 2A JSONL (not recorded as 0); inferred lengths≥64 = 0 | **0** true / 208 false |

Long/repetitive examples appear mainly on 256M (e.g. qid `297` over-generation). 500M max generated length on this set is 22 tokens. Literal reference-substring hits among EM failures are diagnostic only (short answers can match accidentally).

No answer extraction or generation retuning.

## 4. Gallery provenance

Verified all six gallery IDs against pinned DocumentVQA @ `a44195f8…`, manifests, local PNG dimensions/hashes, and saved predictions.

| qid | Correct doc_id / UCSF / page | Image | Provenance |
|-----|------------------------------|-------|------------|
| 5158 | 1762 / `gzyh0227` / 9 | 1666×2149 | OK |
| 292 | 258 / `rzbj0037` / 7 | 2103×1604 | OK |
| 297 | 258 / `rzbj0037` / 7 | 2103×1604 | OK |
| 57368 | 4751 / `snbx0223` / 44 | 1653×2339 | OK (smoke) |
| 5263 | 1785 / `mtnh0227` / 10 | 1704×1105 | OK |
| 53842 | 3200 / `kmfh0023` / 2 | 1692×2245 | OK |

**Correction scope:** Prior gallery docs wrongly showed 5158 as 1565/`yjbj0227`/1 and 5263 as 1585/`yjbj0227`/21. **Documentation only** — manifests and leakage-audit inputs already had the correct fields; no membership change.

Programmatic tables: [`docs/examples/gallery_tables_generated.md`](../examples/gallery_tables_generated.md).
