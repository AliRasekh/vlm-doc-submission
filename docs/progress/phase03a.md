# Phase 3A — InternVL3-1B language_model LoRA (bounded PoC)

**Dates:** 2026-10-02  
**Host:** Neumann  
**Status:** COMPLETED (holdout sealed; no further tuning)

## Hypothesis

DocVQA task adaptation can be tested by freezing the vision encoder + MLP connector and attaching LoRA only under `language_model` attention projections (`q_proj/k_proj/v_proj/o_proj`), keeping the Phase 2C dynamic-448 / `max_num=12` / thumbnail recipe and the same instruction. This tests language-side adaptation; it does **not** claim improved visual resolution.

## Jobs / hardware

| Job | Role | Partition | State | Elapsed | Node / GPU |
|-----|------|-----------|-------|---------|------------|
| **82978** | CPU-only ampere probe | ampere (no GPU) | COMPLETED | ~0s | gpunode02 |
| **82979** | Train-image integrity audit | ampere (no GPU) | COMPLETED | 00:06:59 | gpunode02 |
| **82985** | Gate + 200-update train + dual dev eval | ampere + A100 | COMPLETED 0:0 | **00:39:35** | gpunode02 / **A100-PCIE-40GB** |

Peak train alloc ≈ **5.88 GiB**. Software: torch `2.5.1+cu118`, transformers `4.48.3`, peft `0.14.0` (verified unchanged core stack).

## Image integrity (frozen membership)

Audit: `results/train_subset_v1_image_integrity_audit.json`  
- 2000/2000 OK; **0 repairs**; leakage recheck clean (UCSF/qid/page/src/RGB/PNG).  
- `examples_content_sha256` = `e013cda1…60f0a8`  
- `manifest_raw_file_sha256` = `79fe9aca…502c60d0`  
PNG resume in `prepare_train_subset_v1.py` now requires RGB-hash match before reuse.

## LoRA placement (verified)

- Wrapped subtree: **`language_model` only** via PEFT `get_peft_model`.  
- Pre-wrap Linear targets: **96** modules, all `language_model.model.layers.*.self_attn.{q,k,v,o}_proj`.  
- Trainable tensors: LoRA A/B under `language_model` only (asserted).  
- Total unique params incl. adapters: **940,355,712** (<1e9)  
- Trainable: **2,162,688** (≈0.230%)

## Correctness gate

`results/internvl_lora_gate_ok.json` — **ok**  
- Label masking + IMG_CONTEXT count checks (median + large image).  
- Finite loss; nonzero LoRA grads; frozen param unchanged.  
- Memorization 8×30: loss **0.531 → 0.0026**; save/reload logits max-abs **0.0**, greedy match.  
- Diagnostic adapters discarded; main run reloaded fresh base+LoRA.

Deviation note: torch checkpoint emitted `None of the inputs have requires_grad=True` warnings during training despite `enable_input_require_grads`; gate still showed nonzero LoRA grads and train loss decreased (**0.522 → 0.201** over 200 updates). No OOM; tile budget unchanged.

## Training exposure

- 200 optimizer updates; microbatch 1 × accum 8.  
- **1600** sample presentations; **1600** unique question IDs (not a full pass over 2000).  
- Checkpoints: `checkpoints/internvl3_1b_lora_v1/update_0100`, `update_0200` (gitignored, persistent under repo).  
  - u100 `adapter_model.safetensors` SHA256 `16564ec14d2d2a071f927534a84960ed43f0f0d194ec2d5fdc0bf7d48f67c219`  
  - u200 `adapter_model.safetensors` SHA256 `4b925a8528ce9fddbe3357db6d0223ea2b3547fb37c5380d0924d2bdbd36e1e7`  
- Curves: `results/train_internvl3_1b_lora_v1.jsonl` + `…_curves.json`.  
- Supervised example doc: `docs/examples/train_lora_v1_supervised_example.md`.

## Development results (primary ANLS strict v2; n=208; 0 failures)

| Step | ANLS | EM |
|-----:|-----:|---:|
| 0 baseline (job 82020 preds) | **0.8324** | 0.7692 |
| 100 LoRA | 0.8238 | 0.7692 |
| 200 LoRA | 0.8195 | 0.7644 |

**Selection (predeclared):** step **0** wins (FT did not improve development selection).  
**Best trained among 100/200:** **100**.  
Vs baseline at u100: improved **4** / regressed **4** / unchanged **200**.  
Examples: `docs/examples/lora_v1_before_after.md`.

## Non-goals honored

No holdout inference; no hyperparameter sweep; no prompt/preprocess retuning; baseline JSONL preserved.

## Commands (exact)

```bash
export PYTHONNOUSERSITE=1
sbatch scripts/slurm/audit_train_subset_images.sbatch   # CPU audit
sbatch scripts/slurm/phase03a_internvl_lora.sbatch       # GPU gate+train+eval
```
