# Phase 2C — full review packet + InternVL3-1B development baseline

**Date:** 2026-09-30  
**Host:** Neumann  
**Status:** COMPLETED (job 82020; prior attempt 82018 failed on missing `timm`/`einops`)

## Goals

1. Export a real Phase 2B **full source** review packet (not hash-only summaries).
2. Correct diagnostic-EM documentation (formatting sensitivity / rank reversal).
3. Add **OpenGVLab/InternVL3-1B** as a cross-family third development baseline on the fixed 208-qid set.
4. Compare three models under the same primary scorer; no holdout / no fine-tuning.

## Review packet (Phase 2B full)

Programmatic export: `scripts/export_full_review_packet.py`  
Path: `results/phase02b_full_review.txt` (gitignored)  
Embeds actual source files with `BEGIN_FILE`/`END_FILE` delimiters and a byte inventory.

## Diagnostic EM wording (fix)

Terminal-period diagnostic (secondary): after normalize, strip ≤1 trailing `.` from pred and each ref.

| Model | Primary EM | Diagnostic EM |
|-------|------------|---------------|
| 256M | 54/208 | 103/208 |
| 500M | 9/208 | 131/208 |

This shows **substantial formatting sensitivity** and a **reversed ranking** under the limited diagnostic. It does not make diagnostic EM official accuracy, and does not claim every mismatch is punctuation.

## InternVL3-1B

| Field | Value |
|-------|-------|
| Model | `OpenGVLab/InternVL3-1B` |
| Revision | `4415a3b810e636d11dfa86b0e9ba40bb00535aa8` |
| License | Apache-2.0 (Hub) |
| Architecture | InternVLChatModel = InternVision + mlp connector + Qwen2 LM |
| Measured unique params | **938,193,024** (&lt; 1e9) |
| Preprocess | Official dynamic 448px tiles, `max_num=12`, `use_thumbnail=True`; 256 visual tokens/tile |
| FlashAttention | disabled (`use_flash_attn=false`) |
| Env | Existing `vlm_doc`; added `timm==1.0.30`, `einops==0.8.2` after 82018 ImportError |
| Training data | Hub tags cite MMPR-v1.2 / Instruct base; broader mixture **unknown** beyond public card |

### Job / hardware

| Field | Value |
|-------|-------|
| Job | **82020** COMPLETED 0:0 in 00:03:27 on `gpunode06` |
| GPU | NVIDIA **A100-SXM4-40GB** |
| Smoke | limit=4 first (separate JSONL); then full 208 |
| Timing scope | **`model.generate()` only** + CUDA sync (comparable field to SmolVLM generate timing; unmatched hardware still applies) |
| Generate return | `generated_only` (**verified** Phase 2D job 82527; was labeled `generated_only_assumed` in 2C JSONL) |
| Peak alloc / reserved | ~3.40 / ~6.62 GiB |
| Gen-token median / max / cap hits | 6 / 28 / **0** |
| Preds ending `.` | 8 / 208 |

## Three-model development table (provisional; not holdout)

| Model | Params | ANLS strict v2 | Primary EM | Diagnostic EM | Med gen latency (s) | GPU |
|-------|-------:|---------------:|-----------:|--------------:|--------------------:|-----|
| SmolVLM-256M | 256,484,928 | 0.5568 | 0.2596 | 0.4952 | 0.209 | A100-PCIE-40GB (81847) |
| SmolVLM-500M | 507,482,304 | 0.6095 | 0.0433 | 0.6298 | 0.243 | A100-SXM4-40GB (82001) |
| InternVL3-1B | 938,193,024 | **0.8324** | **0.7692** | 0.7692 | 0.468 | A100-SXM4-40GB (82020) |

Visual budgets are **not matched** (SmolVLM native splitting vs InternVL 448/`max_num=12`/thumbnail). Latency not ranked across unmatched scopes/hardware beyond noting InternVL generate-only median on SXM4.

Machine-readable: `results/dev_three_model_comparison.json`.

## Failure patterns / limitations

- InternVL still errs on some entities/numbers (regressions vs 500M exist).
- SmolVLM-500M primary EM remains depressed by terminal periods; diagnostic EM reverses that ranking.
- Development ranking is **provisional** and **not** final holdout performance.
- No prompt tuning, holdout inference, or fine-tuning in this phase.

## Artifacts

- Adapter: `src/vlm_doc/adapters/internvl.py`
- Config: `configs/models/internvl3_1b.yaml`
- Slurm: `scripts/slurm/dev_internvl3_1b_neumann.sbatch`
- Failure evidence: `logs/vlm_doc_dev_internvl_82018.err` (missing timm/einops)
