#!/usr/bin/env python3
"""Export visual-dependence review packet (no credentials / bulk preds / weights)."""

from __future__ import annotations

import hashlib
import subprocess
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
    out = REPO / "companion/pascal/artifacts/visual_dependence_review.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    include = [
        "docs/companion/pascal/VISUAL_DEPENDENCE_PROTOCOL.md",
        "docs/companion/pascal/VISUAL_DEPENDENCE_RESULTS.md",
        "docs/companion/pascal/VISUAL_DEPENDENCE_VERIFICATION.md",
        "docs/companion/pascal/PLAN.md",
        "docs/companion/pascal/PROGRESS.md",
        "docs/companion/pascal/DECISIONS.md",
        "companion/pascal/README.md",
        "companion/pascal/visual_dep/__init__.py",
        "companion/pascal/visual_dep/paths.py",
        "companion/pascal/visual_dep/freeze_mapping.py",
        "companion/pascal/visual_dep/runner.py",
        "companion/pascal/visual_dep/analyze.py",
        "companion/pascal/visual_dep/gallery.py",
        "companion/pascal/visual_dep/export_review.py",
        "companion/pascal/scripts/visual_dependence.sbatch",
        "companion/pascal/tests/test_visual_dep_mapping.py",
        "companion/pascal/tests/test_visual_dep_analyze.py",
        "companion/pascal/results/review_closeout_cpu_tests.txt",
        "companion/pascal/results/visual_dependence_table.json",
        "companion/pascal/results/visual_dependence_gallery.json",
        "companion/pascal/results/figures/visual_dependence_anls_em.png",
        "companion/pascal/artifacts/visual_dep/unrelated_mapping_meta.json",
        "companion/pascal/artifacts/visual_dep/runtime.json",
        "companion/pascal/artifacts/visual_dep/main_done.json",
        "companion/pascal/artifacts/visual_dep/cond_clean/dev_eval_summary.json",
        "companion/pascal/artifacts/visual_dep/cond_white/dev_eval_summary.json",
        "companion/pascal/artifacts/visual_dep/cond_unrelated/dev_eval_summary.json",
        # reused engine
        "src/vlm_doc/adapters/internvl.py",
        "src/vlm_doc/metrics.py",
        "configs/models/internvl3_1b.yaml",
    ]
    # Truncate mapping body in export — include meta + sha only plus small head via meta
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# PASCAL VISUAL-INPUT DEPENDENCE REVIEW PACKET",
        f"# export_timestamp_utc={ts}",
        f"# final_commit={git_sha()}",
        f"# repo_root={REPO}",
        "# EXCLUDED: credentials, bulk prediction jsonl, adapter/model weights, full unrelated mapping body.",
        "",
        "## FILE_INVENTORY",
    ]
    mapping = REPO / "companion/pascal/artifacts/visual_dep/unrelated_mapping.json"
    if mapping.is_file():
        lines.append(
            f"OK_HASH_ONLY companion/pascal/artifacts/visual_dep/unrelated_mapping.json "
            f"bytes={mapping.stat().st_size} sha256={sha256_file(mapping)}"
        )
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
        if p.suffix.lower() in {".png", ".jpg", ".svg"}:
            lines += [
                "=" * 80,
                f"BEGIN_BINARY path={rel} bytes={p.stat().st_size} sha256={sha256_file(p)}",
                f"END_BINARY path={rel}",
                "",
            ]
            continue
        data = p.read_text(errors="replace")
        lines += [
            "=" * 80,
            f"BEGIN_FILE path={rel} bytes={p.stat().st_size}",
            "=" * 80,
            data if data.endswith("\n") else data + "\n",
            f"END_FILE path={rel}",
            "",
        ]
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    print(f"Wrote {out} sha256={hashlib.sha256(text.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
