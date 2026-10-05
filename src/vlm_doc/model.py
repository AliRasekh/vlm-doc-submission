"""Model loading helpers for SmolVLM / Idefics3-style checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


@dataclass
class LoadedModel:
    model: Any
    processor: Any
    device: torch.device
    dtype: torch.dtype
    revision: str
    model_id: str
    total_parameters: int
    bf16_supported: bool


def count_total_parameters(model: torch.nn.Module) -> int:
    """Count unique parameter tensors (handles tied weights)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += p.numel()
    return total


def load_vlm(
    model_id: str,
    revision: str,
    *,
    device: torch.device | None = None,
    prefer_bf16: bool = True,
) -> LoadedModel:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bf16_supported = False
    if device.type == "cuda":
        bf16_supported = bool(torch.cuda.is_bf16_supported())

    use_bf16 = prefer_bf16 and device.type == "cuda" and bf16_supported
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()

    total = count_total_parameters(model)
    if total >= 1_000_000_000:
        raise AssertionError(
            f"Model total unique parameters {total} exceeds 1e9 budget"
        )

    return LoadedModel(
        model=model,
        processor=processor,
        device=device,
        dtype=dtype if device.type == "cuda" else torch.float32,
        revision=revision,
        model_id=model_id,
        total_parameters=total,
        bf16_supported=bf16_supported,
    )
