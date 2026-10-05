"""Explicit P2 runtime selection and BF16 provenance probes."""

from __future__ import annotations

from typing import Any

import torch


def probe_bf16_support() -> dict[str, Any]:
    """Record usability vs native-support booleans without changing precision."""
    out: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "api": "torch.cuda.is_bf16_supported(including_emulation: bool = True)",
        "default_includes_emulation": True,
    }
    if not torch.cuda.is_available():
        out["is_bf16_supported_default"] = False
        out["is_bf16_supported_including_emulation_false"] = False
        out["device_name"] = None
        out["compute_capability"] = None
        return out

    props = torch.cuda.get_device_properties(0)
    out["device_name"] = torch.cuda.get_device_name(0)
    out["compute_capability"] = [int(props.major), int(props.minor)]
    out["total_memory_bytes"] = int(props.total_memory)
    out["is_bf16_supported_default"] = bool(torch.cuda.is_bf16_supported())
    try:
        out["is_bf16_supported_including_emulation_false"] = bool(
            torch.cuda.is_bf16_supported(including_emulation=False)
        )
    except TypeError:
        out["is_bf16_supported_including_emulation_false"] = None
        out["including_emulation_kw_unsupported"] = True

    # Interpret without claiming kernel paths.
    cc_major = int(props.major)
    out["native_cc_ge_8"] = cc_major >= 8
    out["interpretation"] = {
        "bf16_tensor_usability_default_true": out["is_bf16_supported_default"],
        "native_hardware_support_indicated": bool(
            out.get("is_bf16_supported_including_emulation_false")
        ),
        "kernel_path_verified": False,
        "note": (
            "Default True on CC<8 typically means emulation/usability branch in "
            "PyTorch 2.5.1, not native Ampere BF16. Do not infer acceleration."
        ),
    }
    return out


def p2_dtype_policy() -> dict[str, Any]:
    """Retain the successfully executed P1 dtype (bfloat16) with accurate labels.

    Does not auto-switch precision from the misleading default boolean.
    Not claimed performance-optimal.
    """
    return {
        "selected_dtype": "bfloat16",
        "prefer_bf16": True,
        "use_flash_attn": False,
        "attention_backend_label": "eager",
        "rationale": (
            "Retain P1 job-117225 executed runtime (bfloat16 + eager) for demo continuity. "
            "On V100 this is BF16 tensor usability; native CC>=8 support is separate and "
            "must be recorded via including_emulation=False."
        ),
        "performance_optimal_claim": False,
        "auto_switch_from_support_boolean": False,
    }


def inspect_attention_impl(model: Any) -> dict[str, Any]:
    """Best-effort attention implementation labels (may be incomplete)."""
    info: dict[str, Any] = {"vision": {}, "language": {}, "notes": []}
    try:
        cfg = getattr(model, "config", None)
        info["model_config_attn_implementation"] = getattr(
            cfg, "_attn_implementation", getattr(cfg, "attn_implementation", None)
        )
    except Exception as exc:  # noqa: BLE001
        info["notes"].append(f"config_probe_failed:{exc}")

    # Language model (Qwen2)
    try:
        lm = getattr(model, "language_model", None)
        if lm is not None:
            lcfg = getattr(lm, "config", None)
            info["language"]["attn_implementation"] = getattr(
                lcfg, "_attn_implementation", getattr(lcfg, "attn_implementation", None)
            )
            # First layer module class name
            layers = getattr(getattr(lm, "model", lm), "layers", None)
            if layers is not None and len(layers) > 0:
                attn = getattr(layers[0], "self_attn", None)
                info["language"]["first_layer_self_attn_class"] = (
                    type(attn).__name__ if attn is not None else None
                )
    except Exception as exc:  # noqa: BLE001
        info["notes"].append(f"language_probe_failed:{exc}")

    try:
        vit = getattr(model, "vision_model", None)
        if vit is not None:
            vcfg = getattr(vit, "config", None)
            info["vision"]["attn_implementation"] = getattr(
                vcfg, "_attn_implementation", getattr(vcfg, "attn_implementation", None)
            )
            info["vision"]["class"] = type(vit).__name__
    except Exception as exc:  # noqa: BLE001
        info["notes"].append(f"vision_probe_failed:{exc}")

    info["flash_attn_requested"] = False
    info["label"] = "eager_unless_module_reports_otherwise"
    return info
