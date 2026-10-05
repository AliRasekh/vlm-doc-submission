"""Thin InternVL load wrapper with documented V100 dtype fallback."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from vlm_doc.adapters.internvl import LoadedInternVL, load_internvl


def load_internvl_p1(
    model_id: str,
    revision: str,
    *,
    prefer_bf16: bool = True,
    use_flash_attn: bool = False,
    max_num: int = 12,
    use_thumbnail: bool = True,
    fallback_dtype_if_no_bf16: str = "float16",
) -> tuple[LoadedInternVL, dict[str, Any]]:
    """Load via shared ``load_internvl``, then apply float16 if bf16 unsupported.

    Does not modify model weights beyond dtype cast; no quantization / offload.
    """
    loaded = load_internvl(
        model_id,
        revision,
        prefer_bf16=prefer_bf16,
        use_flash_attn=use_flash_attn,
        max_num=max_num,
        use_thumbnail=use_thumbnail,
    )
    provenance: dict[str, Any] = {
        "loader": "vlm_doc.adapters.internvl.load_internvl",
        "prefer_bf16_requested": bool(prefer_bf16),
        "use_flash_attn": bool(use_flash_attn),
        "attention_backend": "eager" if not use_flash_attn else "flash_attn",
        "bf16_supported": bool(loaded.bf16_supported),
        "dtype_after_load": str(loaded.dtype).replace("torch.", ""),
        "dtype_fallback_applied": False,
        "fallback_dtype_if_no_bf16": fallback_dtype_if_no_bf16,
    }
    if prefer_bf16 and loaded.bf16_supported:
        provenance["resolved_dtype_policy"] = "bfloat16_as_neumann"
        return loaded, provenance

    # Unsupported bf16 (typical V100): documented float16 fallback.
    fb = str(fallback_dtype_if_no_bf16).lower()
    if fb in ("float16", "fp16", "half"):
        target = torch.float16
        name = "float16"
    elif fb in ("float32", "fp32"):
        provenance["resolved_dtype_policy"] = "float32_no_fallback"
        return loaded, provenance
    else:
        raise ValueError(f"unsupported fallback_dtype_if_no_bf16={fallback_dtype_if_no_bf16}")

    if loaded.dtype != target:
        model = loaded.model.to(dtype=target)
        loaded = replace(loaded, model=model, dtype=target)
        provenance["dtype_fallback_applied"] = True
        provenance["resolved_dtype_policy"] = f"{name}_fallback_no_bf16"
    else:
        provenance["resolved_dtype_policy"] = f"{name}_already"

    provenance["dtype_after_fallback"] = str(loaded.dtype).replace("torch.", "")
    return loaded, provenance


def set_max_num(loaded: LoadedInternVL, max_num: int) -> LoadedInternVL:
    """Override only the dynamic tile cap on an already-loaded model handle."""
    return replace(loaded, max_num=int(max_num))
