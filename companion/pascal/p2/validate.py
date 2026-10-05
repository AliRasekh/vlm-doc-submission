"""Input validation for P2 demo (no silent oversized resize)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

# Documented limits (reject rather than silently resize beyond native preprocess intent).
MAX_UPLOAD_BYTES = 12 * 1024 * 1024  # 12 MiB
MAX_PIXELS = 40_000_000  # width*height
ALLOWED_CAPS = (1, 4, 12)


@dataclass
class ValidationError(Exception):
    message: str

    def __str__(self) -> str:  # noqa: D105
        return self.message


def validate_question(question: str | None) -> str:
    if question is None or not str(question).strip():
        raise ValidationError("Question is empty. Enter a non-empty question.")
    q = str(question).strip()
    if len(q) > 4000:
        raise ValidationError("Question exceeds 4000 characters.")
    return q


def validate_cap(max_num: int | str | None) -> int:
    try:
        cap = int(max_num)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError("Tile cap must be an integer in {1, 4, 12}.") from exc
    if cap not in ALLOWED_CAPS:
        raise ValidationError(
            f"Unsupported tile cap {cap}. Allowed: {list(ALLOWED_CAPS)} "
            "(12=default; 4 and 1=reduced visual budgets)."
        )
    return cap


def validate_image_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ValidationError(f"Image file not found: {p}")
    size = p.stat().st_size
    if size <= 0:
        raise ValidationError("Image file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise ValidationError(
            f"Image file exceeds upload-size limit ({MAX_UPLOAD_BYTES} bytes)."
        )
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.load()
            w, h = im.size
            mode = im.mode
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValidationError(f"Unreadable image: {exc}") from exc
    if w <= 0 or h <= 0:
        raise ValidationError("Image has invalid dimensions.")
    if w * h > MAX_PIXELS:
        raise ValidationError(
            f"Image exceeds pixel limit ({MAX_PIXELS} = width*height). "
            "Resize externally before upload; demo does not silently downscale."
        )
    return {
        "path": str(p),
        "bytes": int(size),
        "width": int(w),
        "height": int(h),
        "mode": mode,
        "limits": {"max_upload_bytes": MAX_UPLOAD_BYTES, "max_pixels": MAX_PIXELS},
    }
