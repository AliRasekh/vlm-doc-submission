"""Singleton InternVL engine with serialized GPU requests (no P1 memory monkey-patch)."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vlm_doc.adapters.internvl import load_internvl, run_one_internvl

from .runtime import inspect_attention_impl, p2_dtype_policy, probe_bf16_support
from .validate import validate_cap, validate_image_file, validate_question

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INSTRUCTION = "Answer the question using a single word or short phrase."
MODEL_ID = "OpenGVLab/InternVL3-1B"
REVISION = "4415a3b810e636d11dfa86b0e9ba40bb00535aa8"


class DemoEngine:
    """Load once; serialize all inference; isolate max_num per request."""

    def __init__(self, *, cache_dir: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._loaded = None
        self._load_meta: dict[str, Any] = {}
        self._cache_dir = Path(
            cache_dir or os.environ.get("HF_HOME", REPO_ROOT / ".hf_cache")
        )
        self.instruction = DEFAULT_INSTRUCTION
        self.max_new_tokens = 64
        self.model_id = MODEL_ID
        self.revision = REVISION

    @property
    def ready(self) -> bool:
        return self._loaded is not None

    def _configure_hf_env(self) -> None:
        cache = self._cache_dir.resolve()
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache)
        os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HUB_CACHE"])

    def _load_unlocked(self) -> dict[str, Any]:
        if self._loaded is not None:
            return dict(self._load_meta)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for P2 demo inference")
        self._configure_hf_env()
        policy = p2_dtype_policy()
        bf16_probe = probe_bf16_support()
        loaded = load_internvl(
            self.model_id,
            self.revision,
            prefer_bf16=bool(policy["prefer_bf16"]),
            use_flash_attn=bool(policy["use_flash_attn"]),
            max_num=12,
            use_thumbnail=True,
        )
        param_dtype = None
        for _, p in loaded.model.named_parameters():
            if torch.is_floating_point(p):
                param_dtype = str(p.dtype).replace("torch.", "")
                break
        self._loaded = loaded
        self._load_meta = {
            "model_id": self.model_id,
            "revision": self.revision,
            "dtype": str(loaded.dtype).replace("torch.", ""),
            "param_dtype_sample": param_dtype,
            "bf16_supported_flag_from_loader": bool(loaded.bf16_supported),
            "bf16_probe": bf16_probe,
            "dtype_policy": policy,
            "attention": inspect_attention_impl(loaded.model),
            "attention_backend_label": policy["attention_backend_label"],
            "use_flash_attn": False,
            "instruction": self.instruction,
            "max_new_tokens": self.max_new_tokens,
            "gpu_name": torch.cuda.get_device_name(0),
            "label_runtime": (
                "bfloat16 tensors as executed in P1; native BF16 hardware support "
                "is indicated only by is_bf16_supported(including_emulation=False) "
                "and CC>=8 — not by dtype alone."
            ),
            "memory_metrics_in_demo": False,
            "memory_note": (
                "P2 does not use P1's global reset_peak_memory_stats monkey-patch; "
                "live UI omits memory peaks."
            ),
        }
        return dict(self._load_meta)

    def ensure_loaded(self) -> dict[str, Any]:
        with self._lock:
            return self._load_unlocked()

    def details(self) -> dict[str, Any]:
        with self._lock:
            if self._loaded is None:
                return {"ready": False}
            return {"ready": True, **self._load_meta}

    def predict(
        self,
        *,
        image_path: str | Path,
        question: str,
        max_num: int = 12,
    ) -> dict[str, Any]:
        q = validate_question(question)
        cap = validate_cap(max_num)
        img_meta = validate_image_file(image_path)

        with self._lock:
            self._load_unlocked()
            assert self._loaded is not None
            # Per-request cap isolation; restore stored handle to default 12 after.
            handle = replace(self._loaded, max_num=cap)
            if handle.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            image = Image.open(img_meta["path"]).convert("RGB")
            out = run_one_internvl(
                handle,
                image=image,
                question=q,
                max_new_tokens=self.max_new_tokens,
                instruction=self.instruction,
            )
            if handle.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            self._loaded = replace(self._loaded, max_num=12)

            info = out.get("image_info") or {}
            return {
                "status": "ok",
                "prediction": out.get("prediction"),
                "question": q,
                "max_num_cap": cap,
                "actual_tile_count": int(info.get("tile_count", -1)),
                "thumbnail_appended": bool(info.get("thumbnail_appended", False)),
                "original_size_wh": list(image.size),
                "request_seconds": float(t1 - t0),
                "request_timing_scope": (
                    "image_open_through_preprocess_transfer_generate_decode_"
                    "cuda_synchronized"
                ),
                "generation_seconds": out.get("generation_seconds"),
                "generation_timing_scope": out.get("timing_scope"),
                "generated_token_length": out.get("generated_token_length"),
                "generation_cap_reached": out.get("generation_cap_reached"),
                "instruction": self.instruction,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "model_id": self.model_id,
                "revision": self.revision,
                "dtype": self._load_meta.get("dtype"),
                "attention_backend_label": self._load_meta.get(
                    "attention_backend_label"
                ),
                "gpu_name": self._load_meta.get("gpu_name"),
                "image_meta": img_meta,
                "live_inference": True,
                "references_used_in_model": False,
            }


_ENGINE: DemoEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> DemoEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = DemoEngine()
        return _ENGINE


def save_upload_bytes(data: bytes, *, suffix: str = ".png") -> Path:
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, prefix="p2_upload_"
    )
    try:
        tmp.write(data)
        tmp.flush()
    finally:
        tmp.close()
    return Path(tmp.name)
