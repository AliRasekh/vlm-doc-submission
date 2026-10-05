# Phase 2D — InternVL decode verification + training-subset prep

**Date:** 2026-10-01 / 2026-10-02  
**Host:** Neumann  
**Status:** COMPLETED (stop before fine-tuning / holdout)

## Goals

1. Replace InternVL `generated_only_assumed` with evidenced return convention.
2. Verify / correct Phase 2B–2C full review packets.
3. Prepare a leakage-checked ~2000-example training subset from pinned DocVQA **train**.
4. Preliminary descriptive error sample (≤20) from InternVL development predictions.
5. Docs + regenerate review packet; no training loop / no holdout inference.

## 1. Generate-return verification

### Static trace (pinned remote code + transformers 4.48.3)

- Adapter calls `InternVLChatModel.generate(pixel_values, input_ids, attention_mask, …)`.
- Remote `generate` builds `inputs_embeds` (text embeds with ViT features spliced at `<IMG_CONTEXT>`), then calls `language_model.generate(**inputs_embeds=…)**` — **no `input_ids`** to the LM generate.
- HF `GenerationMixin._maybe_initialize_input_ids_for_generation`: when only `inputs_embeds` are present, starter `input_ids` is shape `(batch, 0)`.
- Returned `LongTensor` is therefore **newly generated token IDs only**.
- Official `chat()` `batch_decode`s that tensor with **no** `input_len` slice (then splits on conversation `sep`).

### Slurm probe

| Field | Value |
|-------|-------|
| Job | **82527** COMPLETED on `gpunode05` (A100-PCIE-40GB) |
| Examples | development qids **5158**, **292** (no reference answers in inference) |
| Artifact | `results/internvl_generate_return_probe.json` |

Evidence per example: `input_token_length` ≫ `returned_token_length`; returned IDs do **not** start with prompt IDs; zero `<IMG_CONTEXT>` in return; adapter prediction **equals** official `chat()` under identical pixels + generation settings.

**Verified convention:** `generated_only`  
**Slicing rule:** decode all returned IDs (no prompt slice); then strip conversation `sep` if present.  
**Baseline action:** **retain** existing InternVL development scores (ANLS 0.8324 / EM 0.7692). Adapter field renamed from `generated_only_assumed` → `generated_only` (historical JSONL still shows the old label; scores unchanged).

## 2. Review packets

| Packet | Bytes | SHA256 | Notes |
|--------|------:|--------|-------|
| `results/phase02b_full_review.txt` | 327029 | `ee647de2b5d36738a769b5748d62eae130b43d84868af0f39af593fe716c7563` | Historical 2026-09-30T12:59:46Z @ `203b51f` |
| `results/phase02c_full_review.txt` (pre-regen) | 327029 | `5aef359af6bf0dd2e8743cb0048c3c14180c82f6a29375495bb05434d4cda029` | Differed from phase02b by **1 byte** (header label only) — same snapshot |
| `results/phase02c_full_review.txt` (**regenerated**) | ~1.93 MiB / 64 files | (local gitignored; recompute `sha256sum`) | Embeds Phase 2D sources + probe/train/error artifacts |
| `results/phase02d_full_review.txt` | same inventory as regen phase02c | (differs by header label only) | Label `phase02d` |

Provenance of the old pair: both exported **2026-09-30T12:59:46Z** at git `203b51f`, dirty=clean. Content differed by **exactly one byte** (`phase02b` vs `phase02c` in the header). Same on-disk source snapshot — **intentional duplicate snapshot, not two independent inventories**. Regenerated packets embed actual current adapter/runner/metrics/train-subset/probe sources (not hashes alone).

## 3. Training subset `docvqa_train_subset_v1`

| Field | Value |
|-------|-------|
| Source | `HuggingFaceM4/DocumentVQA` revision `a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4` **train** (38 shards) |
| Size | **2000** QA examples |
| Seed | **42** |
| Target rule | `first_nonempty_source_listed_answer` (all answers kept in local sidecar) |
| Manifest | `data/manifests/docvqa_train_subset_v1.json` |
| Manifest SHA256 | `79fe9acaa0fc6ed35759a06789d99c338a579929f3de7fd17782322b502c60d0` |
| Images / answers | `data/images/docvqa_train_subset_v1/` , `data/cache/docvqa_train_subset_v1_answers.json` (gitignored) |
| Audit | `results/train_subset_v1_leakage_audit.json` — **leakage_detected=false** |
| Documented example | `docs/examples/train_subset_v1_example.md` (qid 10196; **not** an eval gallery item) |

### Sampling (deterministic)

1. Metadata-scan all train shards.  
2. Drop empty-answer rows.  
3. Exclude qid / UCSF / exact page identity present in smoke, development, or holdout_v2.  
4. Group by UCSF else `PAGE_FALLBACK(doc_id, page, source_image_sha256)` (missing UCSF never collapsed).  
5. Lexicographic group keys → `random.Random(42).shuffle` → take groups; within group sort by `question_id` until target (+buffer).  
6. Stream images; exclude source/RGB/PNG hash collisions with eval sets.  
7. **Independent of model predictions.**

### Leakage-audit counts (selected)

- Eval blocklist: 160 UCSF / 720 qids / 185 pages / 185 src / 185 rgb hashes.  
- Exclusions during scan: **2406** rows with UCSF in eval splits.  
- Overlap after selection: **0** on all signals.  
- Selected: 298 UCSF groups; 0 PAGE_FALLBACK in the final 2000.

**No training loop / no fine-tuning in this phase.**

## 4. Preliminary InternVL error sample

- Seed **42**; **20** of **48** non-perfect / 208 development predictions.  
- Artifacts: `results/internvl_error_sample_v1.json`, `docs/progress/phase02d_error_sample.md`.  
- Observed (image-opened): wrong table/field cells (e.g. 47045, 45817); text-reading/entity (48178, 8120); formatting near-matches (15334, 59706/7, 61509); ambiguous annotation/handwriting (56431, 62533). Uncertain cases marked explicitly.  
- Descriptive only — **no** metric/answer changes.

## Non-goals (honored)

Fine-tuning, holdout evaluation, prompt/preprocess tuning, changing development membership or primary metrics.
