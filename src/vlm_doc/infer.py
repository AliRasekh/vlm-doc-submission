"""Single-example document VQA inference."""

from __future__ import annotations

import time
from typing import Any

import torch
from PIL import Image

from .model import LoadedModel

DEFAULT_INSTRUCTION = (
    "Answer the question using a single word or short phrase."
)


def build_messages(question: str, instruction: str = DEFAULT_INSTRUCTION) -> list[dict]:
    prompt = f"{question.strip()}\n{instruction}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def run_one(
    loaded: LoadedModel,
    *,
    image: Image.Image,
    question: str,
    max_new_tokens: int = 64,
    instruction: str = DEFAULT_INSTRUCTION,
) -> dict[str, Any]:
    """Greedy generation for one image-question pair. No OCR or reference answers."""
    messages = build_messages(question, instruction=instruction)
    prompt = loaded.processor.apply_chat_template(
        messages, add_generation_prompt=True
    )

    original_size = image.size  # (W, H)
    inputs = loaded.processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
    )
    inputs = {k: v.to(loaded.device) for k, v in inputs.items()}

    pixel_values = inputs.get("pixel_values")
    processed_shape = list(pixel_values.shape) if pixel_values is not None else None

    input_len = int(inputs["input_ids"].shape[-1])

    # Image tiling / token diagnostics when available
    image_info: dict[str, Any] = {
        "original_size_wh": list(original_size),
        "processed_pixel_values_shape": processed_shape,
    }
    if "pixel_attention_mask" in inputs:
        image_info["pixel_attention_mask_shape"] = list(
            inputs["pixel_attention_mask"].shape
        )
    if hasattr(loaded.processor, "image_processor"):
        ip = loaded.processor.image_processor
        for attr in (
            "size",
            "do_image_splitting",
            "max_image_size",
            "image_size",
            "do_resize",
        ):
            if hasattr(ip, attr):
                val = getattr(ip, attr)
                try:
                    image_info[f"processor_{attr}"] = val
                except Exception:
                    image_info[f"processor_{attr}"] = str(val)

    if loaded.device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        generated = loaded.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    if loaded.device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    gpu_mem: dict[str, Any] = {}
    if loaded.device.type == "cuda":
        # Peak *PyTorch allocator* stats for this device since the last reset.
        # These are not nvidia-smi total device memory, not host MaxRSS, and not
        # necessarily the full process footprint (CUDA context / fragmentation).
        gpu_mem = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "scope": (
                "torch.cuda.max_memory_{allocated,reserved} since reset_peak_memory_stats "
                "immediately before this generate() call"
            ),
        }

    new_tokens = generated[0, input_len:]
    prediction_raw = loaded.processor.decode(
        new_tokens, skip_special_tokens=True
    )
    prediction = prediction_raw.strip()
    max_new = int(max_new_tokens)
    gen_len = int(new_tokens.numel())

    return {
        "prediction": prediction,
        "prediction_raw": prediction_raw,
        "prediction_stripped": prediction,
        "input_token_length": input_len,
        "generated_token_length": gen_len,
        "generation_cap_reached": gen_len >= max_new,
        "generation_seconds": t1 - t0,
        "image_info": image_info,
        "instruction": instruction,
        "max_new_tokens": max_new,
        "do_sample": False,
        "dtype": str(loaded.dtype).replace("torch.", ""),
        "gpu_memory": gpu_mem,
    }
