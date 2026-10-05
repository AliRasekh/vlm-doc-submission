# Final holdout evaluation protocol (FROZEN)

**Status:** FROZEN for Phase 4 execution. Holdout evaluation is authorized only under this frozen text.  
**Development-selected deployment candidate:** original InternVL3-1B (**step 0**).  
**Fixed adaptation comparison:** InternVL3-1B + corrected LoRA **v2 update 200**.  
**Do not** select a different checkpoint from holdout results.

## Evaluation revision

Recorded at commit time in `results/phase04_evaluation_revision.json` (Git SHA of this frozen protocol + code). That SHA is the evaluation revision.

## Four systems (exact identities)

| Tag | Config | Model ID | Revision | Adapter | Adapter SHA256 |
|-----|--------|----------|----------|---------|----------------|
| smolvlm256m | `configs/models/smolvlm_256m.yaml` | `HuggingFaceTB/SmolVLM-256M-Instruct` | `7e3e67edbbed1bf9888184d9df282b700a323964` | none | — |
| smolvlm500m | `configs/models/smolvlm_500m.yaml` | `HuggingFaceTB/SmolVLM-500M-Instruct` | `a7da5b986cb59b408707209984f360a5f4ad7e47` | none | — |
| internvl3_1b | `configs/models/internvl3_1b.yaml` | `OpenGVLab/InternVL3-1B` | `4415a3b810e636d11dfa86b0e9ba40bb00535aa8` | none | — |
| internvl3_1b_lora_v2_u200 | same InternVL config | same | same | `checkpoints/internvl3_1b_lora_v2/update_0200` | `3c2c34cd751138df0a2a42bfebf90a9a9c7f64b89f2ebc8efe7f5517c344180a` |

Never load LoRA v1 or v2 update 100 for this protocol.

### Checkpoint-selection history (development; frozen)

| Phase | Adaptation comparison | Dev selection |
|-------|----------------------|---------------|
| 3A (v1 labels; historical) | LoRA v1 u100 best trained | Step 0 wins |
| 3C (corrected termination) | LoRA v2 u200 best corrected trained | Step 0 wins |

## Data

- Manifest: `data/manifests/docvqa_holdout_v2.json` — **504** unique question IDs.
- Canonical content SHA256: `e20d9ad6ca9523c36a444faa1d5985f311938d197693c5701d80523b8c50c468`
- Raw file SHA256: `59b845f3cb058e70b96b269966c5eb4dc383409cf7ec0a6fb07ad882430d7dfb`
- Answers sidecar (scoring only): `data/cache/docvqa_holdout_v2_answers.json`
- Preflight verified: 504/504 images exist and match recorded PNG hashes; no train/dev/holdout UCSF, page, QID, or image-hash overlaps under current manifests.

## Prompts and decoding

- Instruction (all four): `Answer the question using a single word or short phrase.`
- `max_new_tokens=64`, `do_sample=false` (greedy), eval/inference mode.
- References never enter model input or generation decisions.

## Native image preprocessing (not matched across families)

| Family | Recipe |
|--------|--------|
| SmolVLM | Native processor image-splitting / resize settings from the model processor (unchanged). |
| InternVL3 | Official dynamic preprocess: 448px tiles, `max_num=12`, thumbnail enabled. |

Visual-token budgets differ. This is a comparison of **configured systems**, not an isolated architecture experiment.

## Scoring

| Role | Metric |
|------|--------|
| **Primary ANLS** | `anls_normalized_strict_v2` (lowercase+strip+collapse; score = 1−NL if NL &lt; 0.5 else 0; max over nonempty refs) |
| **Primary EM** | Normalized exact match (same normalize; no punctuation stripping) |
| **Secondary ANLS** | `anls_vlmevalkit` (explicitly labeled secondary / compatibility) |
| **Secondary diagnostic EM** | After normalize, strip ≤1 trailing ASCII `.` from pred and each ref (formatting sensitivity only; **not** official accuracy) |

- Denominator: **requested-set** means over all 504 QIDs.
- Failures / missing → score 0 for that example in requested means.
- Do not modify reference answers after seeing predictions.

## Confidence interval (InternVL base vs LoRA v2 u200 only)

Paired UCSF-cluster bootstrap on adapted−base primary ANLS:

- Resample complete `ucsf_document_id` groups **with replacement**.
- Retain all questions within each draw, **including multiplicity** when a group is drawn more than once.
- Question-weighted means; difference = adapted − base.
- **B = 2000**, seed **42**, percentile **95%** interval (2.5 / 97.5).

Interval describes uncertainty under this sampled evaluation design, not all sources of uncertainty.

## Execution

- Launcher: `scripts/slurm/final_holdout_eval.sbatch`
- Partition/account/QOS: ampere / ampere / normal; one A100 allocation; models sequential; release each before the next.
- Env: `PYTHONNOUSERSITE=1`, pinned `vlm_doc` conda env (torch 2.5.1+cu118, transformers 4.48.3, peft 0.14.0).
- Separate prediction JSONLs per system; resume only fingerprint-compatible rows.
- Timing: generation-only, CUDA-synchronized; warm-up excluded from latency stats.

## Outputs

- `results/holdout_smolvlm256m.jsonl` + `_summary.json`
- `results/holdout_smolvlm500m.jsonl` + `_summary.json`
- `results/holdout_internvl3_1b.jsonl` + `_summary.json`
- `results/holdout_internvl3_1b_lora_v2_u200.jsonl` + `_summary.json`
- `results/holdout_internvl_anls_bootstrap.json`
- `results/holdout_final_analysis.json` (+ figures under `docs/report/`)

## Non-goals

No further training, HPO, prompt/metric/data changes, or holdout-based checkpoint selection.
