#!/usr/bin/env python3
"""Thin wrapper: invoke scripts/train_internvl_lora.py with P3 seed configs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from . import SEEDS
from .paths import REPO_ROOT, assert_under_p3_artifacts, seed_dirs
from .provenance import write_order_digest


def write_seed_config(seed: int, *, max_updates: int | None = None) -> Path:
    base = yaml.safe_load(
        (REPO_ROOT / "companion/pascal/configs/p3_lora_v2_base.yaml").read_text()
    )
    d = seed_dirs(seed)
    d["root"].mkdir(parents=True, exist_ok=True)
    base["seed"] = int(seed)
    base["checkpoint_root"] = str(d["checkpoint_root"].relative_to(REPO_ROOT))
    base["log_jsonl"] = str(d["log_jsonl"].relative_to(REPO_ROOT))
    base["summary_json"] = str(d["summary_json"].relative_to(REPO_ROOT))
    base["supervised_example_md"] = str(
        d["supervised_example_md"].relative_to(REPO_ROOT)
    )
    base["supervised_example_json"] = str(
        d["supervised_example_json"].relative_to(REPO_ROOT)
    )
    if max_updates is not None:
        base["max_updates"] = int(max_updates)
        base["save_updates"] = []
    # Safety: all destructive paths under artifacts/p3
    for key in (
        "checkpoint_root",
        "log_jsonl",
        "summary_json",
        "supervised_example_md",
        "supervised_example_json",
    ):
        assert_under_p3_artifacts(base[key], kind=key)
    # Never point at original v1/v2 trees
    for forbidden in (
        "checkpoints/internvl3_1b_lora_v1",
        "checkpoints/internvl3_1b_lora_v2",
        "results/train_internvl3_1b_lora_v1",
        "results/train_internvl3_1b_lora_v2",
    ):
        blob = json.dumps(base)
        if forbidden in blob:
            raise RuntimeError(f"config references forbidden path {forbidden}")
    cfg_path = d["root"] / "train_config.yaml"
    cfg_path.write_text(yaml.safe_dump(base, sort_keys=False))
    return cfg_path


def run_train(seed: int, *, max_updates: int | None = None) -> int:
    if seed not in SEEDS and max_updates is None:
        # allow preflight seeds only when max_updates overridden
        pass
    cfg_path = write_seed_config(seed, max_updates=max_updates)
    write_order_digest(seed)
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/train_internvl_lora.py"),
        "--config",
        str(cfg_path),
        "--cache-dir",
        env.get("HF_HOME", str(REPO_ROOT / ".hf_cache")),
        "--skip-gate-marker",
        str(REPO_ROOT / "companion/pascal/artifacts/p3/gate_ok.json"),
    ]
    print("P3_TRAIN_CMD", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Override for preflight timing only",
    )
    args = parser.parse_args()
    return run_train(int(args.seed), max_updates=args.max_updates)


if __name__ == "__main__":
    # allow `python -m` style via path insert
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
