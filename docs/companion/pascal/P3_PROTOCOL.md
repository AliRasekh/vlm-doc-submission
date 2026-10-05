# P3 protocol — fixed-budget three-seed LoRA reproducibility (development only)

**Status:** FROZEN before main training  
**Experiment revision:** `9470055e3c0bfae06a24fad984a6640ddacd1ff0`  
**Label:** Supplementary development analysis designed **after** the original final holdout evaluation.  
**Does not** change deployment selection (InternVL step 0) or official Phase 4 claims.  
**No holdout inference; no holdout answer access; no best-seed selection.**

## Question

How variable is the corrected fixed-budget adaptation outcome across **three training seeds** in one Pascal runtime?

## Seeds (concrete; recorded before training)

| Role | Seed |
|------|-----:|
| Original corrected v2 seed **S** | **42** (from `configs/train/internvl3_1b_lora_v2.yaml`) |
| S+1 | **43** |
| S+2 | **44** |

Only training randomness (PyTorch/Python RNG + data shuffle order) changes between runs. Hyperparameters, objective, membership pool, and target mapping are fixed.

## Model and data

| Item | Value |
|------|-------|
| Base model | `OpenGVLab/InternVL3-1B` |
| Revision (weights + tokenizer) | `4415a3b810e636d11dfa86b0e9ba40bb00535aa8` |
| Train manifest | `data/manifests/docvqa_train_subset_v1.json` (2000 examples) |
| Train answers | `data/cache/docvqa_train_subset_v1_answers.json` |
| Target field / rule | `target_answer` / `first_nonempty_source_listed_answer` |
| Frozen QID→target map | `companion/pascal/artifacts/p3/frozen_qid_target_map.json` (written before runs; identical across seeds) |
| Dev evaluation | `data/manifests/docvqa_dev_v1.json` (208 Q); answers `data/cache/docvqa_dev_v1_answers.json` (scoring only) |

Target selection in the original trainer does **not** depend on RNG (manifest/`target_answer` + answers sidecar). The frozen map records that identity for provenance.

## Training settings (identical to corrected v2 except seed + output paths)

| Setting | Value |
|---------|------:|
| Optimizer updates | **200** (fixed reported checkpoint) |
| Save updates | `{200}` for comparison (also save 100 for diagnostics; **report uses 200 only**) |
| LoRA | language_model `q/k/v/o_proj`; r=16; α=32; dropout=0.05; bias=none |
| LR / WD / clip | 5e-5 / 0.01 / 1.0 |
| Microbatch / accum | 1 / 8 → 8 presentations per update |
| Warmup updates | 10 (linear schedule) |
| max_seq_length | 8192 |
| gradient_checkpointing | true |
| Instruction | `Answer the question using a single word or short phrase.` |
| Preprocess | max_num=12, use_thumbnail=true, image_size=448 |
| Collator fix | `single_assistant_eos_no_trailing_newline` (v2 corrected termination) |
| Attention | `use_flash_attn=false` (eager) |
| prefer_bf16 | **true** (original policy; no automatic FP16 switch) |

Presentations per completed run: `200 × 8 = 1600` microbatch presentations from the 2000-example pool (shuffle+wrap). Different seeds may consume different 1600-subsets/orders. This studies **overall training-seed variability** (init + order/membership), not initialization alone.

## Runtime policy

- One GPU at a time; prefer V100 if original BF16 training path passes the compatibility gate.
- Same physical GPU for all three main runs within one allocation when practical; **fresh Python process per seed**.
- Do not compare runs with different precision/backends as seed-only variation.
- Record: GPU name, CC, `is_bf16_supported()` default and `including_emulation=False`, param/compute dtype, attention impl.
- Do **not** treat BF16 usability as native acceleration without measurement.

## Compatibility gate (required before main training)

Separate diagnostic process using `scripts/gate_internvl_lora.py` + P3 train config. Checks include:

- dtype / attention provenance and BF16 support flags  
- exactly one supervised terminal EOT; no trailing supervised newline  
- intact answer tokens; masked prompt/image tokens  
- trainable = LoRA adapters only; frozen vision/connector  
- finite loss; nonzero adapter grads; frozen weights unchanged  
- small memorization gate (loss decreases)  
- save/reload logits within documented tolerance (`5e-2`) + greedy string match  

Diagnostic adapters/optimizer/RNG **must not** initialize main runs. Marker: `companion/pascal/artifacts/p3/gate_ok.json`.

If the original BF16 training policy cannot pass on the selected hardware: **stop** dependent training; document evidence and a concrete compatible-hardware alternative.

## Evaluation (development only)

Systems (same allocation preferred):

1. Fresh original base (no adapter)  
2. Seed 42 adapter @ update 200  
3. Seed 43 adapter @ update 200  
4. Seed 44 adapter @ update 200  

Settings: max_num=12, thumbnail on, frozen instruction, greedy, max_new_tokens=64, primary `anls_normalized_strict_v2` + EM, requested-set denominator (208). References never enter generation. Separate fingerprints/output files per system (include adapter hash).

Compare fresh Pascal base vs prior P1 cap-12 control diagnostically (no isolated-answer chasing).

## Interrupted / failed runs

- Infrastructure interruptions may retry with **identical** settings and explicit provenance.  
- Do **not** claim exact optimizer-state resume from adapter-only checkpoints.  
- If a complete restart is required, document it and preserve interrupted evidence under `artifacts/p3/`.  
- Do not launch extra seeds to replace disappointing results.

## Analysis (descriptive; three seeds only)

Per seed: seed, adapter hash, completed updates, presentations, unique QIDs, first/last 20-update mean loss, dev ANLS/EM, ΔANLS vs fresh base, improved/regressed/tied counts, failures/missing, gen-cap hits.

Across three Pascal adapted ANLS values: arithmetic mean, sample SD (ddof=1), min, max. Show all three points. **No** population-significance claim; **no** best-seed selection.

Historical Neumann v2 u200 is a **separate reference** — excluded from mean/SD.

## Outputs (tracked vs gitignored)

| Tracked | Gitignored |
|---------|------------|
| Compact tables/figures under `companion/pascal/results/` | `companion/pascal/artifacts/p3/` (logs, adapters, raw preds, gate) |
| Protocol/results/verification docs | |

## Reproduction commands (outline)

```bash
export PYTHONNOUSERSITE=1
# freeze targets (CPU)
$HOME/.conda/envs/vlm_doc_pascal_p1/bin/python companion/pascal/p3/freeze_targets.py
# GPU: gate then main (see sbatch scripts)
sbatch companion/pascal/scripts/p3_preflight.sbatch
# after gate ok:
sbatch companion/pascal/scripts/p3_main.sbatch
```
