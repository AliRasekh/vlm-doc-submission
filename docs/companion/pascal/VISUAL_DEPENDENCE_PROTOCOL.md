# Visual-input dependence — protocol (development only)

**Status:** FROZEN before inference  
**Label:** Post-holdout **supplementary development** analysis.  
**Does not** change deployment selection (InternVL step 0) or official Phase 4 claims.  
**No holdout access. No training. No LoRA adapters.**

## Question

How much does the original selected **InternVL3-1B base** model's development performance depend on receiving the correct document image?

## Systems / conditions (fixed; not adapted to outcomes)

Evaluate **all 208** development questions under three conditions in **one** GPU allocation:

| ID | Condition | Image input |
|----|-----------|-------------|
| `clean` | Correct original image | Manifest `image_relpath` as-is |
| `white` | White image | Solid white RGB with **exactly** the original width×height |
| `unrelated` | Unrelated document image | Deterministic donor from the **same** development split |

## Fixed runtime / decoding (must match verified Pascal base eval)

- Model: `OpenGVLab/InternVL3-1B` @ `4415a3b810e636d11dfa86b0e9ba40bb00535aa8`
- Instruction: `Answer the question using a single word or short phrase.`
- Greedy (`do_sample=false`), `max_new_tokens=64`
- `max_num=12`, `use_thumbnail=true`, official dynamic preprocess
- `prefer_bf16=true`, `use_flash_attn=false`
- Manifest: `data/manifests/docvqa_dev_v1.json` (208)
- Answers sidecar used **only for scoring**, never as generation input or donor selection feature

## Unrelated-image mapping (frozen before inference)

- Seed: **2026**
- For each query QID, select one donor QID from the same 208-set such that:
  - different `doc_id` (and different `ucsf_document_id` when both present);
  - different page identity (`ucsf_document_page_no` when present; else page fallback via image identity);
  - different image content hash (`image_sha256_png`);
- Prefer documented **aspect-ratio** match: minimize `|log(w/h)_query − log(w/h)_donor|`, break ties by ascending donor QID, then apply a seeded shuffle only for residual ties if needed (implementation records the exact rule).
- **Do not** resize donors to imitate original geometry.
- Record donor dimensions and actual tile counts at inference time.
- Freeze mapping JSON + SHA256 under `companion/pascal/artifacts/visual_dep/`.

### Clarification / executed tie-break (protocol deviation; mapping not regenerated)

The original bullet above describes tie-breaking as ascending donor QID first, with seeded shuffle only for residual ties.  
**Actual executed code** (`companion/pascal/visual_dep/freeze_mapping.py`, seed 2026, frozen mapping used by job 117260):

1. Collect eligible donors; sort candidates by `(aspect_log_distance, donor_qid)`.
2. Take the minimum aspect-distance set.
3. If that set has one donor → use it (`unique_min_aspect_distance`).
4. If multiple donors share the same minimum aspect distance → **`random.Random(2026).choice(...)` among those equal-distance donors** (candidates already ordered by QID before the tied set is formed). Recorded as `seeded_choice_among_equal_aspect_distance`.

This is a documentation/protocol-vs-implementation deviation only. The frozen mapping and inference outputs are **preserved**; do **not** regenerate the mapping or rerun inference for this clarification.

**Interpretation of unrelated:** scores are against the **original question's references**. This may also change layout and visual token/tile count; it is a **mismatch diagnostic**, not a pure content-only intervention.

## Metrics (requested-set denominator)

For each condition, report:

- requested / ok / failed / missing, duplicates, generation-cap hits
- strict ANLS v2 and EM over all 208 requested QIDs (failures/missing = 0; **generation-cap cases remain included** in these means)
- paired ANLS Δ vs `clean`
- normalized prediction agreement with `clean`: count matches **only** when both condition and clean records have `status=ok`; denominator = number of such comparable pairs (missing/error pairs are excluded and must not agree via empty-string normalization). Agreement is **not** correctness against references.
- tile-count distribution (report what was **observed** in the run; white×original geometry may match clean tiles in a given run without claiming that as a general law)

## Non-claims

- Residual correct answers under white/unrelated do **not** establish memorization or contamination.
- This experiment does **not** prove fine-grained visual grounding.
- A large ANLS drop under white/unrelated indicates strong dependence on correct visual input; it is **not** an exact decomposition of performance.
- Not the planned severity-robustness (P4) study.

## Overlap policy

Reuse prior Pascal outputs only when **complete provenance** matches this protocol (same model revision, instruction, decoding, tile policy, manifest, and condition semantics). Otherwise re-run. Clean/white/unrelated for this study are produced in one allocation for comparability.
