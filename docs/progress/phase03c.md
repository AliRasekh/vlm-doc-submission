# Phase 3C — Correct supervised end-of-turn + repeat LoRA once

**Dates:** 2026-10-02  
**Host:** Neumann  
**Status:** COMPLETED (holdout sealed; no further training)  
**Job:** **83333** COMPLETED 00:42:07 on gpunode04 / **A100-PCIE-40GB**

## Bug (verified)

In v1 `build_supervised_example`, after tokenizing the full conversation, the builder appended `eos_id` whenever `full_ids[-1] != eos_id`.

InternVL `internvl2_5` template `sep` is `"<|im_end|>\n"`. The assistant turn therefore already ends with `<|im_end|>` **plus** a newline token (`Ċ`). Because the last id was the newline, v1 appended a **second** `<|im_end|>`.

Evidence (qid **46350**, Phase 3A packet `results/train_lora_v1_supervised_example.json`):

```
... submission . <|im_end|> Ċ <|im_end|>
```

Token before/after: `results/supervised_termination_qid46350_before_after.json`.  
V2 excerpt ends with a single `<|im_end|>` (`results/train_lora_v2_supervised_example.json`).

**Affected count** (text tokenization only, no image materialization): **2000 / 2000** train-subset examples would append the second eos (`results/supervised_termination_audit_v1.json`).

### Why the Phase 3A gate missed it

The gate checked label-mask excerpts, IMG_CONTEXT counts, finite loss, LoRA grads, memorization, and save/reload, but did **not** assert “exactly one supervised end-of-turn and no supervised trailing newline.” The duplicated terminator still yields finite/decreasing loss, so those checks passed.

This phase does **not** claim the bug caused the v1 development ANLS regression; it corrects the objective and remeasures.

## Correction

- Locate the assistant terminator **inside the assistant suffix** (after the inference prompt prefix).
- Keep answer tokens + **exactly one** `<|im_end|>`; drop trailing newline / second eos.
- Retain exact prompt-prefix tokenization assertion; no silent answer truncation.
- Regression tests: `tests/test_collator_termination.py` (11 tests, OK).
- Gate updated with `termination_single_eos` check (`results/internvl_lora_v2_gate_ok.json`).

## V1 preservation

V1 checkpoints/preds/robustness/docs retained. V1 adaptation is **superseded for the intended answer+one-EOT objective** but kept as historical observed results with the implementation caveat. Base step 0 remains valid. Training unlinks only the configured log path; v2 refuses to unlink a v1 log (`results/phase03c_v1_log_guard.json` verified).

## Training (v2; recipe unchanged except labels)

- Config: `configs/train/internvl3_1b_lora_v2.yaml`
- 200 updates; 1600 unique QIDs; peak ≈ 5.87 GiB
- Loss (per-update accum mean): **0.253 → 0.301**; first-20 mean = **0.25614097751677034**; last-20 mean = **0.21429072171449662** (documentation correction: earlier ≈0.268/≈0.207 were stale approximations vs the packaged loss array) 
  (`docs/examples/phase03c_train_loss_curve_v2.svg`)
- Checkpoints: `checkpoints/internvl3_1b_lora_v2/update_{0100,0200}`
  - u100 SHA256 `6f563aa4adb47b9f…81584e14`
  - u200 SHA256 `3c2c34cd751138df…c344180a`

## Development selection (n=208; primary ANLS strict v2)

| Step | ANLS | EM |
|-----:|-----:|---:|
| 0 baseline | **0.8324** | 0.7692 |
| v2 u100 | 0.8263 | 0.7692 |
| v2 u200 | 0.8290 | 0.7740 |

**Selected for deployment:** step **0** (FT still does not improve selection).  
**Best corrected trained:** **200**.  
Vs baseline at u200: 6 improved / 4 regressed / 198 unchanged.

### V1 vs v2 (bug-correction comparison, not causal proof)

| | v1 best trained (100) | v2 best trained (200) |
|--|----------------------:|----------------------:|
| ANLS | 0.8238 | 0.8290 |
| EM | 0.7692 | 0.7740 |

Compact: `results/phase03c_v1_vs_v2_dev.json`.

## Robustness refresh (best v2 = u200; frozen 54-Q / 16-UCSF subset)

Fresh clean vs full-dev v2 u200 on subset QIDs: **54/54** exact match.

| Model | Condition | ANLS | EM | Note |
|-------|-----------|-----:|---:|------|
| base | clean | 0.7671 | 0.7037 | Phase 3B reused |
| base | half_detail | 0.7691 | 0.7222 | Phase 3B reused |
| base | jpeg_q40 | 0.7292 | 0.6667 | Phase 3B reused |
| v1 lora100 | clean | 0.7535 | 0.7222 | **historical** (buggy labels) |
| v1 lora100 | half_detail | 0.7700 | 0.7222 | historical |
| v1 lora100 | jpeg_q40 | 0.7144 | 0.6667 | historical |
| **v2 lora200** | clean | **0.7708** | 0.7222 | corrected |
| v2 lora200 | half_detail | 0.7700 | 0.7222 | corrected |
| v2 lora200 | jpeg_q40 | 0.7315 | 0.6852 | corrected |

`results/robustness_v2_analysis.json`. Tiny deltas ≠ statistical superiority.

## Final protocol

Updated to use **LoRA v2 update 200** as the fixed adaptation comparison; v1 selection history preserved in docs. Holdout **not** executed.

## Non-goals honored

No hyperparameter search; no holdout inference; no claim that duplicated termination alone explains every v1↔v2 score difference.
