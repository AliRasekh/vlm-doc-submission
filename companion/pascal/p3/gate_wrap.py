#!/usr/bin/env python3
"""Run the existing InternVL LoRA gate into companion/pascal/artifacts/p3/."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from .paths import REPO_ROOT, assert_under_p3_artifacts
from .train_wrap import write_seed_config


def probe_runtime() -> dict:
    out: dict = {
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        out["gpu_name"] = torch.cuda.get_device_name(0)
        out["compute_capability"] = list(torch.cuda.get_device_capability(0))
        out["is_bf16_supported_default"] = bool(torch.cuda.is_bf16_supported())
        out["is_bf16_supported_including_emulation_false"] = bool(
            torch.cuda.is_bf16_supported(including_emulation=False)
        )
        out["native_cc_ge_8"] = tuple(torch.cuda.get_device_capability(0)) >= (8, 0)
    return out


def run_gate(out_path: str | Path = "companion/pascal/artifacts/p3/gate_ok.json") -> int:
    out = assert_under_p3_artifacts(out_path, kind="gate_out")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Use seed-42 path layout for gate config (settings only; adapters discarded).
    cfg_path = write_seed_config(42)
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/gate_internvl_lora.py"),
        "--config",
        str(cfg_path),
        "--out",
        str(out),
        "--cache-dir",
        env.get("HF_HOME", str(REPO_ROOT / ".hf_cache")),
    ]
    print("P3_GATE_CMD", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)

    runtime = probe_runtime()
    if out.is_file():
        report = json.loads(out.read_text())
    else:
        report = {"ok": False, "checks": {}}
    report["pascal_p3_runtime"] = runtime
    report["prefer_bf16_policy"] = True
    report["fp16_auto_switch"] = False
    report["gate_config"] = str(cfg_path.relative_to(REPO_ROOT))
    report["dtype_policy_note"] = (
        "Original v2 prefer_bf16=true retained. V100 default is_bf16_supported() "
        "may be True via emulation; including_emulation=False indicates native support."
    )
    if rc != 0:
        report["ok"] = False
    out.write_text(json.dumps(report, indent=2) + "\n")

    if rc != 0:
        print("P3_GATE_FAILED", flush=True)
        return rc
    if not report.get("ok"):
        return 1
    print(
        "P3_GATE_OK",
        json.dumps(
            {
                "gpu": runtime.get("gpu_name"),
                "bf16_default": runtime.get("is_bf16_supported_default"),
                "bf16_native": runtime.get(
                    "is_bf16_supported_including_emulation_false"
                ),
            }
        ),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="companion/pascal/artifacts/p3/gate_ok.json",
    )
    args = parser.parse_args()
    return run_gate(args.out)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
