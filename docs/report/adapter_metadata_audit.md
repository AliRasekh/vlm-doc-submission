# Adapter `base_model_name_or_path` metadata audit

**Scope:** packaging correction only. Original checkpoint weights and `adapter_config.json` on disk are left byte-identical.

## Observed field

In `adapter_config.json` (checkpoint and delivered ZIP):

```text
"base_model_name_or_path": "./pretrained/Qwen2.5-32B-Instruct"
```

This string alone is **not** evidence that a 32B model was trained or adapted.

## Origin (traced)

1. Pinned InternVL3-1B Hub config at revision `4415a3b810e636d11dfa86b0e9ba40bb00535aa8` embeds:

   ```text
   llm_config._name_or_path = "./pretrained/Qwen2.5-32B-Instruct"
   ```

   while the same `llm_config` reports `hidden_size=896`, `num_hidden_layers=24` (consistent with the card’s Qwen2.5-0.5B-scale language tower, not a 32B tower). Vision card name remains InternViT-300M-448px-V2_5; the vision config’s own `_name_or_path` is a separate stale string and is unused by our LoRA path.

2. Training wraps **only** `internvl_model.language_model` with PEFT (`src/vlm_doc/train/lora.py` → `get_peft_model` / `save_pretrained`). PEFT copies the wrapped module’s config `_name_or_path` into adapter `base_model_name_or_path`.

3. Therefore the ZIP field is **inherited upstream metadata** from the language submodule config inside InternVL3-1B, not a project-chosen base-model ID and not a claim that adapters attach to a standalone Qwen2.5-32B checkpoint.

Do **not** rewrite this field to `OpenGVLab/InternVL3-1B`: the adapter wraps the **language submodule**, not the top-level VLM object.

## Supported loading route (explicit; required)

```text
load_internvl("OpenGVLab/InternVL3-1B", revision="4415a3b8…")
  → PeftModel.from_pretrained(internvl_model.language_model, adapter_dir)
```

Implemented in:

- `src/vlm_doc/adapters/internvl.py` (`load_internvl`)
- `src/vlm_doc/train/lora.py` (`load_language_lora`)
- `scripts/run_eval.py` (loads InternVL from YAML `model_id`/`revision`, then optional `--lora-adapter`)

Generic automatic base-model loading that trusts `adapter_config.json`’s `base_model_name_or_path` (e.g. “open this path as the full model”) is **not** the supported route and would be incorrect for this package.

## Checks performed in this packaging pass (CPU)

| Check | Result |
|-------|--------|
| Checkpoint `adapter_model.safetensors` SHA256 | `3c2c34cd…c344180a` |
| ZIP weight SHA256 identical to checkpoint | yes |
| ZIP tensor inventory (safetensors, CPU) | 192 tensors; 96×`lora_A` + 96×`lora_B` |
| Tensor shapes | `(16,896)`, `(896,16)`, `(128,16)` — matches r=16 and hidden size 896 |
| Loading mismatch requiring weight edits | **none found** |

## Historical loading evidence (prior phases; not re-run now)

`results/internvl_lora_v2_gate_ok.json` → `checks.save_reload`: `ok=true`, `logits_max_abs_diff=0.0`, greedy prediction match after save/reload of language LoRA (Phase 3C gate). Holdout job 84636 used the same explicit InternVL+`--lora-adapter` path.

## Package documentation

Packaged ZIP `README.md` and `docs/report/SUBMISSION_GUIDE.md` state the inheritance explanation and the supported load route. Checkpoint files themselves were not modified.
