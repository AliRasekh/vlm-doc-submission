#!/usr/bin/env python3
"""Export untracked P1 review packet (plain text)."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="companion/pascal/artifacts/P1_review.txt",
    )
    parser.add_argument(
        "--experiment-sha",
        default=None,
        help="Protocol/impl commit SHA recorded before full eval",
    )
    args = parser.parse_args()
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    include = [
        "companion/pascal/README.md",
        "companion/pascal/.gitignore",
        "companion/pascal/configs/p1_visual_budget.yaml",
        "companion/pascal/p1/__init__.py",
        "companion/pascal/p1/caps.py",
        "companion/pascal/p1/load_wrap.py",
        "companion/pascal/p1/infer_wrap.py",
        "companion/pascal/p1/fingerprint.py",
        "companion/pascal/p1/runner.py",
        "companion/pascal/p1/analyze.py",
        "companion/pascal/p1/export_review.py",
        "companion/pascal/tests/test_p1_caps.py",
        "companion/pascal/scripts/p1_visual_budget.sbatch",
        "docs/companion/pascal/PLAN.md",
        "docs/companion/pascal/PROGRESS.md",
        "docs/companion/pascal/DECISIONS.md",
        "docs/companion/pascal/P1_PROTOCOL.md",
        "docs/companion/pascal/P1_RESULTS.md",
        "companion/pascal/results/p1_visual_budget_table.json",
        "companion/pascal/results/p1_gallery.json",
        "companion/pascal/results/P1_RESULTS.md",
        "companion/pascal/results/figures/p1_anls_vs_request_latency.svg",
        "companion/pascal/results/figures/p1_anls_vs_tile_usage.svg",
        "companion/pascal/results/figures/p1_memory_summary.svg",
        "companion/pascal/artifacts/p1_runs/runtime.json",
        "companion/pascal/artifacts/p1_runs/run_index.json",
    ]
    # Per-condition summaries (compact)
    runs = REPO / "companion/pascal/artifacts/p1_runs"
    if runs.is_dir():
        for p in sorted(runs.glob("p1_dev_max_num_*_summary.json")):
            include.append(str(p.relative_to(REPO)))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    final_sha = git_sha()
    exp_sha = args.experiment_sha or final_sha

    lines: list[str] = []
    lines.append("# PASCAL P1 REVIEW PACKET")
    lines.append(f"# export_timestamp_utc={ts}")
    lines.append(f"# experiment_revision={exp_sha}")
    lines.append(f"# final_commit={final_sha}")
    lines.append(f"# repo_root={REPO}")
    lines.append(
        "# EXCLUDED: model weights, images, raw prediction jsonl, whole manifests, secrets."
    )
    lines.append("")
    lines.append("## FILE_INVENTORY")
    for rel in include:
        p = REPO / rel
        if p.is_file():
            lines.append(f"OK {rel} bytes={p.stat().st_size} sha256={sha256_file(p)}")
        else:
            lines.append(f"MISSING {rel}")
    lines.append("")

    for rel in include:
        p = REPO / rel
        if not p.is_file():
            continue
        data = p.read_text(errors="replace")
        lines.append("=" * 80)
        lines.append(f"BEGIN_FILE path={rel} bytes={p.stat().st_size}")
        lines.append("=" * 80)
        lines.append(data)
        if not data.endswith("\n"):
            lines.append("")
        lines.append(f"END_FILE path={rel}")
        lines.append("")

    text = "\n".join(lines) + "\n"
    out.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    print(f"Wrote {out} size={len(text.encode())} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
