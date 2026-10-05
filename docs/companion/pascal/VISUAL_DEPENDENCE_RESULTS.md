# Visual-input dependence — results (development only)

**Label:** Post-holdout supplementary development analysis.  
**Does not** change deployment selection or official Phase 4 claims.  
**No holdout. No training. Base model only (no P3 adapter).**

## Question

How much does the original selected InternVL3-1B base depend on receiving the correct document image?

## Runtime

| Field | Value |
|-------|-------|
| Job | **117260** COMPLETED (00:25:18) |
| Node / GPU | pascal-node09 / Tesla V100-PCIE-32GB |
| dtype | bfloat16 (`prefer_bf16=true`; BF16 usability, not native CC≥8) |
| Attention | eager (`use_flash_attn=false`) |
| Model | `OpenGVLab/InternVL3-1B` @ `4415a3b810e636d11dfa86b0e9ba40bb00535aa8` |
| Source revision | `5a4c5174feb5331ed641480cd756b81d8654d096` |
| Mapping seed / sha256 | 2026 / `6d45e1b188f89e9a501213889b5381e417e0fa88d602184bd271d2fe34cce674` |
| Conditions | clean → white → unrelated (same allocation; independent requests) |

## Three-condition table (n=208 requested)

Failures/missing count as 0 on the requested-set denominator. Generation-cap hits are **included** in those means (not dropped). No duplicates. Unrelated scores are against the **original** question references.

| Condition | ANLS | EM | ΔANLS vs clean | pred agreement vs clean | ok / fail / miss | gen-cap | mean tiles |
|-----------|-----:|---:|---------------:|------------------------:|-----------------:|--------:|-----------:|
| clean | 0.8276 | 0.7644 | — | 1.0000 | 208 / 0 / 0 | 0 | 12.58 |
| white | 0.0421 | 0.0288 | **−0.7855** | 0.0240 | 208 / 0 / 0 | 2 | 12.58 |
| unrelated | 0.0469 | 0.0288 | **−0.7807** | 0.0288 | 208 / 0 / 0 | 1 | 12.63 |

**Prediction agreement vs clean** = normalized string match to the clean-condition prediction, counted **only** when both records have `status=ok`. Denominator = number of such comparable pairs (here 208/208 for all conditions). It is **not** correctness against references (use ANLS/EM for that). Missing/error pairs are excluded and do not agree via empty-string normalization.

Tile histograms **observed in this run**: clean `{7:13, 10:3, 13:192}`; white `{7:13, 10:3, 13:192}` (**same tile-count distribution as clean in this run**, with white images matching original width×height); unrelated `{7:13, 13:195}`. White–clean tile matching here is an observation for this allocation, not a claimed invariant of all white-image runs.

Paired ANLS vs clean: white 2↑ / 172↓ / 34=; unrelated 2↑ / 170↓ / 36=.

Clean ANLS/EM match the prior P3 fresh-base means on this development set (diagnostic; same model/settings family; this job re-ran clean in-allocation).

## Interpretation

Replacing the correct page with a white or unrelated image reduced development ANLS by approximately **78 percentage points**, indicating **strong dependence on the correct visual input**.

That statement is a coarse summary of the observed drop (≈0.785 / 0.781 ANLS), **not** an exact decomposition of performance into visual vs non-visual parts, **not** proof of fine-grained visual grounding, and **not** evidence for or against contamination/memorization.

- **White:** residual ANLS ≈ 0.04 without readable document content; small residual remains without establishing memorization.
- **Unrelated:** similarly collapsed scores vs the original question’s references; may also change layout/tile structure — a **mismatch diagnostic**, not ordinary accuracy on the donor, and not a pure content-only ablation.

## Artifacts

| Path | Role |
|------|------|
| `companion/pascal/results/visual_dependence_table.json` | Compact metrics |
| `companion/pascal/results/figures/visual_dependence_anls_em.png` | Comparison figure |
| `companion/pascal/results/visual_dependence_gallery.json` + `visual_dependence_gallery/` | Illustrative gallery (not representative) |
| `companion/pascal/artifacts/visual_dep/` | Mapping, runtime, condition jsonl/summaries (gitignored bulk) |

## Limitations

- Development set only; one GPU allocation (V100-PCIE).
- Unrelated donors not resized — tile/layout confound intentional and documented.
- Prior Phase-3B `half_detail`/`jpeg_q40` robustness is a different protocol/subset — not reused.
- P4 severity robustness remains **not started**.
