#!/usr/bin/env python3
"""Evaluate fresh base or a P3 adapter on development via scripts/run_eval.py."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .paths import REPO_ROOT, P3_ART, assert_under_p3_artifacts, seed_dirs


def run_eval(
    *,
    tag: str,
    lora_adapter: str | None,
    train_update: int | None,
    out_jsonl: Path,
    summary_json: Path,
) -> int:
    assert_under_p3_artifacts(out_jsonl, kind="eval_jsonl")
    assert_under_p3_artifacts(summary_json, kind="eval_summary")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_eval.py"),
        "--config",
        "configs/models/internvl3_1b.yaml",
        "--manifest",
        "data/manifests/docvqa_dev_v1.json",
        "--answers",
        "data/cache/docvqa_dev_v1_answers.json",
        "--out-jsonl",
        str(out_jsonl),
        "--summary-json",
        str(summary_json),
        "--score",
        "--cache-dir",
        env.get("HF_HOME", str(REPO_ROOT / ".hf_cache")),
    ]
    if lora_adapter:
        cmd += ["--lora-adapter", lora_adapter]
    if train_update is not None:
        cmd += ["--train-update", str(train_update)]
    print("P3_EVAL", tag, " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=["base", "seed"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.system == "base":
        out_dir = P3_ART / "eval_fresh_base"
        return run_eval(
            tag="fresh_base",
            lora_adapter=None,
            train_update=None,
            out_jsonl=out_dir / "dev_eval.jsonl",
            summary_json=out_dir / "dev_eval_summary.json",
        )

    seed = int(args.seed)
    d = seed_dirs(seed)
    adapter = d["checkpoint_root"] / "update_0200"
    if not adapter.is_dir():
        print(f"ERROR: missing adapter {adapter}", file=sys.stderr)
        return 2
    return run_eval(
        tag=f"seed_{seed}",
        lora_adapter=str(adapter),
        train_update=200,
        out_jsonl=d["eval_jsonl"],
        summary_json=d["eval_summary"],
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
