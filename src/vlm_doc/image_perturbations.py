"""Deterministic development-only image perturbations (Phase 3B).

Transforms are applied to a copy of the clean RGB image. Original files on disk
are never modified. Dimensions at the model-preprocessing boundary match the
clean image size for every condition.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

from PIL import Image

# Fixed resampler for down/up scale path (documented; not score-tuned).
HALF_DETAIL_RESAMPLE = Image.Resampling.BICUBIC

PERTURBATION_IDS = ("clean", "half_detail", "jpeg_q40")


def _sha256_png_bytes(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def apply_perturbation(
    image: Image.Image,
    *,
    perturbation: str,
) -> tuple[Image.Image, dict[str, Any]]:
    """Return transformed RGB image + metadata. ``image`` must already be RGB."""
    if image.mode != "RGB":
        raise ValueError(f"expected RGB image, got {image.mode}")
    orig_w, orig_h = image.size
    pid = str(perturbation)
    if pid not in PERTURBATION_IDS:
        raise ValueError(f"unknown perturbation {pid!r}; expected one of {PERTURBATION_IDS}")

    if pid == "clean":
        out = image.copy()
        meta: dict[str, Any] = {
            "perturbation_id": "clean",
            "settings": {"identity": True},
            "original_size": [orig_w, orig_h],
            "pre_model_size": [orig_w, orig_h],
            "pillow_version": Image.__version__,
            "resample_name": None,
        }
    elif pid == "half_detail":
        # floor(dim/2), min 1; then restore exact original dimensions.
        half_w = max(1, orig_w // 2)
        half_h = max(1, orig_h // 2)
        small = image.resize((half_w, half_h), resample=HALF_DETAIL_RESAMPLE)
        out = small.resize((orig_w, orig_h), resample=HALF_DETAIL_RESAMPLE)
        meta = {
            "perturbation_id": "half_detail",
            "settings": {
                "downscale": "floor_div_2_min1",
                "intermediate_size": [half_w, half_h],
                "restore_original_size": True,
                "resample": "BICUBIC",
                "resample_enum": int(HALF_DETAIL_RESAMPLE),
            },
            "original_size": [orig_w, orig_h],
            "pre_model_size": list(out.size),
            "pillow_version": Image.__version__,
            "resample_name": "BICUBIC",
        }
    else:  # jpeg_q40
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=40, subsampling=2)
        jpeg_bytes = buf.getvalue()
        out = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        if out.size != (orig_w, orig_h):
            raise RuntimeError(
                f"JPEG round-trip changed size {out.size} != {(orig_w, orig_h)}"
            )
        meta = {
            "perturbation_id": "jpeg_q40",
            "settings": {
                "format": "JPEG",
                "quality": 40,
                "subsampling": 2,
                "subsampling_note": "Pillow subsampling=2 (4:2:0)",
            },
            "original_size": [orig_w, orig_h],
            "pre_model_size": list(out.size),
            "jpeg_bytes_sha256": hashlib.sha256(jpeg_bytes).hexdigest(),
            "jpeg_nbytes": len(jpeg_bytes),
            "pillow_version": Image.__version__,
            "resample_name": None,
        }

    if tuple(out.size) != (orig_w, orig_h):
        raise RuntimeError(
            f"perturbation {pid} changed pre-model size {out.size} != {(orig_w, orig_h)}"
        )
    meta["transformed_rgb_png_sha256"] = _sha256_png_bytes(out)
    meta["source_rgb_png_sha256"] = _sha256_png_bytes(image)
    return out, meta
