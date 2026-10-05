"""LoRA attach/save/load restricted to InternVL language_model subtree."""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model

TARGET_LINEAR_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")


def freeze_non_lora_modules(internvl_model: Any) -> None:
    for p in internvl_model.vision_model.parameters():
        p.requires_grad = False
    for p in internvl_model.mlp1.parameters():
        p.requires_grad = False


def attach_language_lora(
    internvl_model: Any,
    *,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    bias: str = "none",
) -> dict[str, Any]:
    """Wrap only ``language_model`` with LoRA on q/k/v/o_proj."""
    freeze_non_lora_modules(internvl_model)

    pre_matches = [
        name
        for name, mod in internvl_model.named_modules()
        if name.split(".")[-1] in TARGET_LINEAR_NAMES
        and isinstance(mod, torch.nn.Linear)
    ]
    bad = [n for n in pre_matches if not n.startswith("language_model.")]
    if bad:
        raise RuntimeError(
            f"Refusing LoRA attach: target Linear names outside language_model: {bad}"
        )
    if not pre_matches:
        raise RuntimeError("No language_model q/k/v/o_proj Linear modules found")

    lora_cfg = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        bias=bias,
        task_type="CAUSAL_LM",
        target_modules=list(TARGET_LINEAR_NAMES),
    )
    internvl_model.language_model = get_peft_model(
        internvl_model.language_model, lora_cfg
    )

    # Ensure vision/connector remain frozen after PEFT wrap.
    freeze_non_lora_modules(internvl_model)

    trainable = [n for n, p in internvl_model.named_parameters() if p.requires_grad]
    bad_train = [
        n
        for n in trainable
        if (not n.startswith("language_model."))
        or ("lora_" not in n and "modules_to_save" not in n)
    ]
    if bad_train:
        raise RuntimeError(
            f"Unexpected trainable parameters: {bad_train[:20]}"
        )

    matched_lora_modules = [
        n
        for n, _m in internvl_model.named_modules()
        if n.startswith("language_model.")
        and any(x in n for x in ("lora_A", "lora_B"))
    ]

    n_total = sum(p.numel() for p in internvl_model.parameters())
    n_trainable = sum(p.numel() for p in internvl_model.parameters() if p.requires_grad)
    if n_total >= 1_000_000_000:
        raise RuntimeError(f"Total params {n_total} exceed 1e9 budget")

    return {
        "pre_wrap_language_linear_targets": pre_matches,
        "matched_lora_parameter_modules": matched_lora_modules,
        "n_matched_lora_modules": len(matched_lora_modules),
        "trainable_param_names": trainable,
        "total_unique_parameters_including_adapters": int(n_total),
        "trainable_parameters": int(n_trainable),
        "trainable_fraction": float(n_trainable) / float(n_total),
        "lora_config": {
            "r": int(r),
            "lora_alpha": int(alpha),
            "lora_dropout": float(dropout),
            "bias": bias,
            "target_modules": list(TARGET_LINEAR_NAMES),
            "wrapped_subtree": "language_model",
        },
    }


def load_language_lora(internvl_model: Any, adapter_dir: str) -> Any:
    freeze_non_lora_modules(internvl_model)
    internvl_model.language_model = PeftModel.from_pretrained(
        internvl_model.language_model, adapter_dir, is_trainable=False
    )
    internvl_model.eval()
    return internvl_model


def save_language_lora(internvl_model: Any, adapter_dir: str) -> None:
    from pathlib import Path

    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    lm = internvl_model.language_model
    if not isinstance(lm, PeftModel):
        raise TypeError("language_model is not a PeftModel")
    lm.save_pretrained(adapter_dir)
