# P1 protocol — InternVL3-1B visual-budget ablation (development only)

**Status:** FROZEN before full evaluation  
**Label:** Supplementary exploratory development analysis performed **after** the original final holdout packaging.  
**Does not** change deployment selection (InternVL step 0) or official Phase 4 claims.  
**No holdout inference; no holdout answer access.**

## Question

How does changing **only** the maximum dynamic image-tile budget (`max_num`) affect development answer quality, inference latency, and GPU memory for the frozen InternVL3-1B **base** model (no LoRA)?

## Model and data

| Item | Value |
|------|-------|
| Model | `OpenGVLab/InternVL3-1B` |
| Revision | `4415a3b810e636d11dfa86b0e9ba40bb00535aa8` |
| Adapter | none |
| Manifest | `data/manifests/docvqa_dev_v1.json` (208 questions; existing membership/order) |
| Answers | `data/cache/docvqa_dev_v1_answers.json` (scoring only; never model input) |
| Cache | `.hf_cache` |

## Native preprocessing (inspected)

Shared implementation: `src/vlm_doc/adapters/internvl.py` (`dynamic_preprocess`, `load_pixel_values`).

| Setting | Value |
|---------|------:|
| `image_size` | 448 |
| `min_num` | 1 |
| Original `max_num` **M** | 12 |
| `use_thumbnail` | true |
| Aspect-ratio selection | official `find_closest_aspect_ratio` over grids with `min_num ≤ i·j ≤ max_num` |
| Normalization | ImageNet mean/std |
| Actual tensors | tile crops **plus** thumbnail when `use_thumbnail` and block count ≠ 1 |

CPU dry-run on all 208 dev images (same algorithm):

| Cap | Observed actual tile counts |
|----:|-----------------------------|
| 1 | always 1 |
| 4 | {3, 5} (thumbnail when blocks ≠ 1) |
| 12 | {7, 10, 13} |

The configured cap is **not** always equal to the tensor count (thumbnail rule).

## Candidate caps

Predeclared unique sorted set: `{1, min(4, M), M}` with `M=12` → **`{1, 4, 12}`**.

- Do **not** increase M.
- Validate vs `min_num=1`; omit illegal caps (none).
- Three distinct settings → run all three.

## What changes / what stays fixed

**Change only:** `max_num` (tile-budget cap).

**Fixed across conditions:**

- Native preprocess algorithm, 448px tile size, normalization, thumbnail rule
- Instruction: `Answer the question using a single word or short phrase.`
- Greedy decoding (`do_sample=false`), `max_new_tokens=64`
- Model weights/revision (base only)
- Runtime dtype + attention backend (identical for all caps)
- Scoring: primary `anls_normalized_strict_v2` + primary EM; requested-set denominator; failures/missing → 0
- Warm-up: development examples at manifest indices `[0,1,2]` before each condition; excluded from scores/latency summaries
- Execution order: **12 → 4 → 1** (fresh control first)
- Single model load; sequential conditions on one physical GPU

## Runtime policy (Pascal / V100)

Neumann reference: `bfloat16`, `use_flash_attn=false`.

On V100, bf16 is unsupported. P1 uses documented **`float16`** fallback with **eager** attention (`use_flash_attn=false`). No quantization, CPU offload, or silent precision emulation. Record actual GPU name/memory, driver, Python, torch/CUDA, transformers, dtype, attention backend.

## Fingerprints

Each condition fingerprints model revision, development manifest content hash, prompt, decoding, preprocessing including **cap**, companion+shared source hashes, software, resolved dtype, attention backend, and experiment source git revision.

Predictions from one cap **must not** resume into another (distinct fingerprints / output files).

## Outputs

| Path | Role |
|------|------|
| `companion/pascal/artifacts/p1_runs/p1_dev_max_num_XX.jsonl` | Raw predictions (gitignored) |
| `companion/pascal/artifacts/p1_runs/*_summary.json` | Per-condition summaries |
| `companion/pascal/results/` | Compact tracked tables/figures |
| `docs/companion/pascal/P1_RESULTS.md` | Narrative results |

## Measurement scopes

- **Generation latency:** shared `run_one_internvl` CUDA-synchronized `model.generate()` only.
- **Request latency:** image open → preprocess → transfer → generate → decode (CUDA sync at boundaries); excludes model load, reference load, scoring.
- **Request peak memory:** peak allocated/reserved over the request window (inner generate-only reset suppressed for that window). Reserved memory reflects allocator caching; not a minimum VRAM floor.
- Filesystem page cache may reduce later-condition request latency.

## Historical comparison

Compare fresh Pascal `max_num=12` predictions to `results/dev_internvl3_1b.jsonl` for exact-response and primary-score agreement. If dtype/hardware differ, do **not** demand equality or attribute differences solely to hardware.

## Analysis rules

- Paired improved/regressed/tied vs fresh Pascal control (`max_num=12`).
- Compact table + SVG figures (ANLS vs latency; ANLS vs tile usage; memory summary).
- Up to four deterministic gallery examples (regression / improvement / unchanged / tile reduction).
- No statistical superiority claims from tiny deltas; no Pascal↔Neumann latency ranking as controlled comparison.

## Resume

Compatible resume: append-only JSONL rows whose `run_fingerprint_sha256` matches the current condition fingerprint. Mixed fingerprints rejected.

## Non-actions

No training, new models, holdout eval, prompt tuning, metric redesign, deployment reselection, P2 execution.
