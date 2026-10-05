"""Request-scoped timing/memory around shared ``run_one_internvl``."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from PIL import Image

from vlm_doc.adapters.internvl import LoadedInternVL, run_one_internvl


@contextmanager
def _hold_peak_memory_across_inner_resets() -> Iterator[None]:
    """Keep peak stats across ``run_one_internvl``'s internal reset.

    Measurement harness only: the shared backend resets peaks immediately before
    ``generate()``. For request-scoped peaks we suppress that reset after an
    outer reset so preprocess + generate share one peak window.
    """
    if not torch.cuda.is_available():
        yield
        return
    real = torch.cuda.reset_peak_memory_stats

    def _noop(*_a: Any, **_k: Any) -> None:
        return None

    torch.cuda.reset_peak_memory_stats = _noop  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.cuda.reset_peak_memory_stats = real  # type: ignore[assignment]


def run_one_request(
    loaded: LoadedInternVL,
    *,
    image_path: str,
    question: str,
    max_new_tokens: int,
    instruction: str,
) -> dict[str, Any]:
    """Open image + shared InternVL inference; request latency excludes load/score."""
    use_cuda = loaded.device.type == "cuda"
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with _hold_peak_memory_across_inner_resets():
        image = Image.open(image_path).convert("RGB")
        out = run_one_internvl(
            loaded,
            image=image,
            question=question,
            max_new_tokens=max_new_tokens,
            instruction=instruction,
        )
        if use_cuda:
            torch.cuda.synchronize()
    t1 = time.perf_counter()

    request_mem: dict[str, Any] = {}
    if use_cuda:
        request_mem = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "scope": (
                "torch.cuda.max_memory_{allocated,reserved} from reset immediately "
                "before image open through synchronized return of run_one_internvl "
                "(inner generate-only reset suppressed for this window)"
            ),
        }

    out = dict(out)
    out["request_seconds"] = float(t1 - t0)
    out["request_timing_scope"] = (
        "image_open_through_preprocess_transfer_generate_decode_cuda_synchronized"
    )
    out["request_gpu_memory"] = request_mem
    out["original_size_wh"] = list(image.size)
    # tile_count from shared preprocess info
    info = out.get("image_info") or {}
    out["actual_tile_count"] = int(info.get("tile_count", -1))
    out["max_num_cap"] = int(info.get("max_num", loaded.max_num))
    out["thumbnail_appended"] = bool(info.get("thumbnail_appended", False))
    return out
