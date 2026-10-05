# Data

## Role

Frozen DocVQA identifier manifests and (gitignored) local images/answers used by inference and scoring.

## Tracked vs local

| Path | Tracked | Contents |
|------|---------|----------|
| `data/manifests/*.json` | yes | IDs, questions, image relpaths, hashes, selection metadata |
| `data/manifests/historical/` | yes | Superseded manifests (do not silently overwrite history) |
| `data/images/` | no | PNG pages for smoke/dev/holdout |
| `data/cache/*_answers.json` | no | Reference answer sidecars (never model inputs) |

## Splits

1. **Smoke (`docvqa_smoke_v1`)** — 8 evenly spaced IDs from validation shard 0 (Phase 1).
2. **Development (`docvqa_dev_v1`)** — **208** fixed question IDs (unchanged since Phase 2A).
3. **Final holdout (`docvqa_holdout_v2`)** — **504** questions, seed 42, grouped by **UCSF source-document ID** (page-level fallback if missing). Supersedes `docvqa_holdout_v1`.
4. **Train subset (`docvqa_train_subset_v1`)** — **2000** questions from pinned **train** split, seed 42, UCSF/page/hash leakage-checked against smoke/dev/holdout_v2. Target = first nonempty source-listed answer. Images/answers gitignored. **No training executed in Phase 2D.**

`docvqa_holdout_v1` is **superseded** (6 UCSF overlaps with development under corrected grouping). Historical copy: `data/manifests/historical/docvqa_holdout_v1.superseded.json`.

No inference on holdout through Phase 2D.

## Grouping policy (Phase 2B correction)

- Prefer `ucsf_document_id` when present and nonempty.
- Missing UCSF → `(PAGE_FALLBACK, doc_id, page, source_image_sha256)` — **never** collapse unknowns into one empty-string group.
- **`doc_id` alone is not a reliable whole-PDF grouping key** (178 UCSF IDs span multiple `doc_id`s in validation).
- Leakage audits (`scripts/audit_split_leakage.py`) **fail explicitly** on UCSF / page / source-hash / RGB-hash overlap between holdout and development/smoke.
- **Future training samples** must be checked against development and holdout UCSF identities and page hashes before use.
- Phase 2D train subset builder: `scripts/prepare_train_subset_v1.py` (audit `results/train_subset_v1_leakage_audit.json`).

## Manifest hashes

`manifest_sha256` is the **canonical content hash** (JSON with `sort_keys`, excluding the self-hash field, plus trailing newline). It is **not** the same as the raw on-disk file SHA256.

## Build commands

```bash
export PYTHONNOUSERSITE=1
# Historical Phase 2A builder (doc_id grouping) — kept for provenance
python scripts/prepare_frozen_splits.py
# Holdout v2 rebuild (UCSF grouping; keeps fixed development IDs)
python scripts/rebuild_holdout_v2.py
python scripts/audit_split_leakage.py --holdout data/manifests/docvqa_holdout_v2.json
```
