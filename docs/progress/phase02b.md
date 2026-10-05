# Phase 2B — corrections + SmolVLM-500M development baseline

**Date:** 2026-09-30  
**Host:** Neumann  
**Status:** COMPLETED (job 82001)

## Goals

1. Correct source-document grouping claims (`doc_id` ≠ reliable whole-PDF key).
2. Freeze primary metric `anls_normalized_strict_v2` (NL < 0.5) and CPU-rescore historical 256M predictions.
3. Harden manifest hash / inference fingerprint / resume provenance.
4. Fix `PYTHONNOUSERSITE` documentation and save `prediction_raw` before strip.
5. Run SmolVLM-500M on the **fixed** 208 development question IDs.
6. Compare under the same v2 metric; update gallery with paired predictions.

## Source-document correction

### Findings (cached validation metadata)

| Fact | Value |
|------|-------|
| Validation rows with missing UCSF | **0** |
| UCSF IDs spanning multiple `doc_id`s | **178** |
| `doc_id`s spanning multiple UCSFs | **0** |
| Development Q IDs (fixed) | **208** |
| Development UCSF source docs | **45** |
| Holdout v1 UCSF source docs | **109** |
| **UCSF overlap development ↔ holdout v1** | **6** (`gfhv0228`, `jjvg0227`, `jrcy0227`, `krcy0227`, `kzng0227`, `ynbx0223`) |
| UCSF overlap smoke ↔ holdout v1 | **0** |
| Exact page-identity / source-hash / RGB-hash overlap dev↔holdout v1 | **0** |
| Smoke↔dev UCSF overlap | **1** (`snbx0223`) — smoke excluded from holdout by construction |

**Conclusion:** `doc_id` does **not** group every page of the same UCSF source PDF. Holdout v1 had **source-document leakage** via UCSF overlap (no page/hash collision).

Gallery counterexample (corrected): qid **5158** = `doc_id=1762`, UCSF `gzyh0227` p.9; qid **5263** = `doc_id=1785`, UCSF `mtnh0227` p.10.

### Holdout v2

- Group by `ucsf_document_id`; missing → `(PAGE_FALLBACK, doc_id, page, source_image_sha256)` (never empty-string collapse).
- Exclude smoke+dev Q IDs and their UCSFs; seed **42**; ~500 Q target; independent of predictions.
- Result: **504** questions, **109** UCSF groups.
- Historical v1 preserved; leakage audit on v2: **0** overlaps (`results/split_leakage_holdout_v2.json`). Audits fail explicitly on overlap.
- **Future training** samples must be checked against development/holdout UCSF IDs and page hashes.

Development membership (**208** Q IDs) unchanged.

## Metric revision

| Metric | Role |
|--------|------|
| `anls_normalized_strict_v2` | **Primary** — 1−NL iff NL < 0.5 else 0 |
| `anls_normalized_inclusive_v1` | Legacy Phase 2A primary |
| `anls_vlmevalkit` | Compatibility |

Not claimed identical to Pix2Struct normalization; strict cutoff cites pix2struct @ `19e41996…` (`docs/sources.md`).

### 256M CPU rescore (no GPU)

| | Inclusive v1 (Phase 2A) | Strict v2 (Phase 2B) |
|--|-------------------------|----------------------|
| Mean ANLS requested | **0.5664** | **0.5568** |
| Exact match | 0.2596 | 0.2596 |
| Cutoff-affected examples | **4** | qids 50537, 52183, 24499, 62533 |

Historical summary: `results/historical/dev_smolvlm256m_summary_phase02a.json`.

## Provenance / resume

- Canonical manifest content hash ≠ raw file SHA256; historical manifests updated with matching content hash + `manifest_sha256_phase02a_recorded` where needed.
- `inference_v2` fingerprint includes inference source hashes (not docs); scoring provenance separate.
- Image content hashes verified before inference; mixed-fingerprint outputs rejected.
- Tests: `tests/test_metrics_v2.py`, `tests/test_fingerprint.py`.

## Environment / raw outputs

- `PYTHONNOUSERSITE=1` required in launcher **before** Python (`scripts/check_env_isolation.py`).
- 500M saves `prediction_raw` + stripped `prediction`, `generated_token_length`, `generation_cap_reached`.
- Prompt / `max_new_tokens=64` unchanged; no preprocess tuning.

## SmolVLM-500M

| Field | Value |
|-------|-------|
| Model | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| Revision | `a7da5b986cb59b408707209984f360a5f4ad7e47` |
| Unique parameters | **507,482,304** (&lt; 1e9) |
| Job | **82001** COMPLETED 0:0 in 00:03:24 on `gpunode06` |
| GPU | **NVIDIA A100-SXM4-40GB** (BF16) — differs from 256M A100-PCIE-40GB |
| Host MaxRSS | ~3.06 GiB (CPU RSS) |
| Peak GPU allocated / reserved | 3.260 / 3.855 GiB |
| Requested / ok / failed | 208 / 208 / 0 |
| Generation cap reached | 0 / 208 |
| Median generated tokens | 6 |

Preprocessor configs at pinned revisions are identical (`longest_edge` 2048 / tile 512 / splitting on). Do **not** assume equal visual-token budgets across models.

## Comparison (same development IDs; primary = strict v2)

| Metric | 256M | 500M |
|--------|------|------|
| ANLS strict v2 | **0.5568** | **0.6095** |
| Exact match | 0.2596 | 0.0433 |
| Gen latency median / p95 (s) | 0.209 / 0.334 | 0.243 / 0.437 |
| Peak alloc / reserved GiB | 2.780 / 3.307 | 3.260 / 3.855 |
| Hardware | A100-PCIE-40GB | A100-SXM4-40GB |

### Secondary (legacy / compatibility)

| Metric | 256M | 500M |
|--------|------|------|
| ANLS inclusive v1 | 0.5664 | 0.6287 |
| ANLS VLMEvalKit | 0.5665 | 0.6287 |

### Per-question (strict v2; point estimates only)

| | Count |
|--|------:|
| Improved (500 > 256) | 53 |
| Regressed (500 < 256) | 70 |
| Tied | 85 |

No statistical significance claimed from these counts alone. 500M ends with `.` on 197/208 predictions vs 98/208 for 256M — likely contributes to the EM drop.

Machine-readable: `results/dev_256_vs_500_comparison.json`.

## CPU follow-up (no new inference)

See [`phase02b_cpu_followup.md`](phase02b_cpu_followup.md): validated paired scores; ΣΔANLS +33.89 / −22.92; terminal-period **diagnostic** EM (256M 103, 500M 131) — not a primary metric; gallery provenance confirmed documentation-only for 5158/5263 errors.

## Non-goals (this phase)

No holdout inference, fine-tuning, prompt/preprocess score-driven tuning, or broad dependency upgrades.
