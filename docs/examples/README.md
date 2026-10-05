# Input/output examples gallery

**Illustrative only — not representative performance evidence.**  
Selected from existing **smoke** and **development** predictions only. The frozen **holdout** was not inspected or evaluated.

Dataset: `HuggingFaceM4/DocumentVQA` revision `a44195f8…` (validation).  
Models: SmolVLM-256M @ `7e3e67ed…` (job 81847, A100-PCIE-40GB); SmolVLM-500M @ `a7da5b98…` (job 82001, A100-SXM4-40GB).

**Phase 2B corrections:** gallery `doc_id`/UCSF fields match manifests (`doc_id` ≠ whole-PDF key). Primary scores use `anls_normalized_strict_v2`.

**Phase 2B/2C:** Programmatic tables: [`gallery_tables_generated.md`](gallery_tables_generated.md). UCSF/doc_id errors for 5158/5263 were **documentation-only**. Paired 256M / 500M / InternVL3-1B predictions for development gallery IDs. Development ranking is provisional — **not** holdout performance.

**Phase 2D training example (separate from evaluation gallery):** [`train_subset_v1_example.md`](train_subset_v1_example.md) — real supervised input/target from `docvqa_train_subset_v1` (qid 10196). Not smoke/dev/holdout.

**Phase 3A / 3B / 3C:** LoRA before/after [`lora_v1_before_after.md`](lora_v1_before_after.md), [`lora_v2_before_after.md`](lora_v2_before_after.md); eight changed examples [`phase03a_eight_changed.md`](phase03a_eight_changed.md); robustness gallery [`robustness_v1_gallery.md`](robustness_v1_gallery.md); corrected supervised example [`train_lora_v2_supervised_example.md`](train_lora_v2_supervised_example.md).

## Selection rules (reproducible)

Original six IDs (Phase 2A follow-up rules on 256M predictions): `5158`, `292`, `297`, `57368`, `5263`, `53842`.

Additional paired callouts (development only; largest |Δ ANLS strict v2| 500M−256M):

- **improvement:** first among sorted improvements by (−Δ, qid) that is already in the gallery if present, else top improvement overall → gallery already includes **297** (Δ = +1.0). Extra top improvement outside gallery: **63666**.
- **regression:** first among sorted regressions by (Δ, qid) → **36943** (Δ = −1.0).

Smoke-only `57368` has **no** 500M prediction (not invented).

---

## Shared model input pattern

```text
{question}
Answer the question using a single word or short phrase.
```

References never enter the model input. 256M historical predictions are **stripped decoded outputs** (no `prediction_raw`). 500M records store `prediction_raw` + stripped `prediction`.

---

## Example A — Correct exact match (`5158`)

| Field | 256M | 500M |
|-------|------|------|
| qid / doc / UCSF | 5158 / 1762 / `gzyh0227` p.9 | same |
| Prediction | `Water Analysis` | `Water Analysis.` |
| ANLS strict v2 / EM | 1.0 / 1.0 | 0.9333 / 0.0 |

Trailing period on 500M drops EM; ANLS remains high.

---

## Example B — Partial / near match (`292`)

| Field | 256M | 500M |
|-------|------|------|
| qid / doc / UCSF | 292 / 258 / `rzbj0037` p.7 | same |
| Prediction | `USMM  1/95-6/95, 12-Month Data.` | `USMM 1/95-6/95, 12-Month Data.` |
| ANLS strict v2 / EM | 0.9667 / 0.0 | 0.9667 / 0.0 |

---

## Example C — Clearly incorrect → improvement (`297`)

| Field | 256M | 500M |
|-------|------|------|
| qid / doc / UCSF | 297 / 258 / `rzbj0037` p.7 | same |
| Question | What is the percentage of Total Menthol in USMM, 1/95-6/95? | |
| References | `78.2%`, `78.2` | |
| Prediction | long over-generation starting `78.2% …` | `78.2%` |
| ANLS strict v2 / EM | 0.0 / 0.0 | **1.0 / 1.0** |

Largest gallery improvement (Δ = +1.0).

---

## Example D — Numeric (`57368`) — smoke only

| Field | Value |
|-------|-------|
| qid / doc / UCSF | 57368 / 4751 / `snbx0223` p.44 |
| Split | smoke |
| 256M | `2` (ANLS/EM 1.0) |
| 500M | **not evaluated** (smoke-only row) |

---

## Example E — Formatting-sensitive (`5263`)

| Field | 256M | 500M |
|-------|------|------|
| qid / doc / UCSF | 5263 / 1785 / `mtnh0227` p.10 | same |
| Prediction | `Mr. Tom Dudley.` | `Mr. tom rudley.` |
| ANLS strict v2 / EM | 0.9333 / 0.0 | 0.8667 / 0.0 |

500M introduces a character error (`rudley`).

---

## Example F — Longer answer (`53842`)

| Field | 256M | 500M |
|-------|------|------|
| qid / doc / UCSF | 53842 / 3200 / `kmfh0023` p.2 | same |
| Prediction | `Promotion overlay is a name generation piece (not a coupon).` | `coupon.` |
| ANLS strict v2 / EM | 0.0 / 0.0 | 0.0 / 0.0 |

---

## Extra paired callouts (selection: top |Δ|)

### Improvement `63666` (Δ = +1.0)

256M: `Salaries.` → 500M: `$ 110,141` (matches reference style; illustrative only).

### Regression `36943` (Δ = −1.0)

256M: `1,028` → 500M: `668.` (exact→wrong under strict v2).

---

## Aggregate note (development, n=208)

Under strict v2, 500M mean ANLS rises (0.5568 → 0.6095) while EM falls sharply (0.2596 → 0.0433), coincident with **197/208** 500M predictions ending in `.` vs **98/208** for 256M. Point estimates only — no significance claimed. Hardware differs (PCIE vs SXM4 A100-40GB); do not treat latency as matched-hardware.

## Latency scope

Generation-only (`model.generate` + CUDA sync); warm-up excluded.
