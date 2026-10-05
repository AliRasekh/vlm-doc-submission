#!/usr/bin/env python3
"""Export P3 review packet (no credentials / bulk preds / manifests)."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="companion/pascal/artifacts/P3_review.txt")
    args = parser.parse_args()
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    include = [
        "docs/companion/pascal/P3_PROTOCOL.md",
        "docs/companion/pascal/P3_RESULTS.md",
        "docs/companion/pascal/P3_VERIFICATION.md",
        "docs/companion/pascal/PLAN.md",
        "docs/companion/pascal/PROGRESS.md",
        "docs/companion/pascal/DECISIONS.md",
        "companion/pascal/README.md",
        "companion/pascal/configs/p3_lora_v2_base.yaml",
        "companion/pascal/p3/__init__.py",
        "companion/pascal/p3/paths.py",
        "companion/pascal/p3/freeze_targets.py",
        "companion/pascal/p3/provenance.py",
        "companion/pascal/p3/train_wrap.py",
        "companion/pascal/p3/gate_wrap.py",
        "companion/pascal/p3/eval_wrap.py",
        "companion/pascal/p3/preflight.py",
        "companion/pascal/p3/run_main.py",
        "companion/pascal/p3/analyze.py",
        "companion/pascal/p3/export_review.py",
        "companion/pascal/scripts/p3_preflight.sbatch",
        "companion/pascal/scripts/p3_main.sbatch",
        "companion/pascal/scripts/p3_main_continue.sbatch",
        "companion/pascal/tests/test_p3_provenance.py",
        "companion/pascal/tests/test_p3_analyze.py",
        "companion/pascal/results/review_closeout_cpu_tests.txt",
        "companion/pascal/results/p3_seed_table.json",
        "companion/pascal/results/p3_verified_analysis.json",
        "companion/pascal/results/p3_loss_curves.json",
        "companion/pascal/results/figures/p3_loss_curves.png",
        "companion/pascal/results/figures/p3_dev_anls.png",
        "companion/pascal/artifacts/p3/gate_ok.json",
        "companion/pascal/artifacts/p3/preflight.json",
        "companion/pascal/artifacts/p3/frozen_qid_target_map.json",
        "companion/pascal/artifacts/p3/main_done.json",
        # reused engine (not modified)
        "scripts/train_internvl_lora.py",
        "scripts/gate_internvl_lora.py",
        "scripts/run_eval.py",
        "src/vlm_doc/train/collator.py",
        "src/vlm_doc/train/lora.py",
        "configs/train/internvl3_1b_lora_v2.yaml",
    ]
    # Per-seed compact summaries + meta (not raw jsonl)
    for seed in (42, 43, 44):
        include += [
            f"companion/pascal/artifacts/p3/seed_{seed:02d}/train_summary.json",
            f"companion/pascal/artifacts/p3/seed_{seed:02d}/order_digest.json",
            f"companion/pascal/artifacts/p3/seed_{seed:02d}/dev_eval_summary.json",
            f"companion/pascal/artifacts/p3/seed_{seed:02d}/checkpoints/update_0200/vlm_doc_checkpoint_meta.json",
        ]
    include.append("companion/pascal/artifacts/p3/eval_fresh_base/dev_eval_summary.json")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# PASCAL P3 REVIEW PACKET",
        f"# export_timestamp_utc={ts}",
        f"# final_commit={git_sha()}",
        f"# repo_root={REPO}",
        "# EXCLUDED: credentials, private keys, whole manifests, images, raw prediction jsonl, adapter weights.",
        "",
        "## FILE_INVENTORY",
    ]
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
        if "BEGIN OPENSSH PRIVATE KEY" in data:
            continue
        # Truncate huge frozen map embedding — keep header fields only if huge
        if rel.endswith("frozen_qid_target_map.json") and len(data) > 200_000:
            obj_head = data[:5000] + "\n... TRUNCATED_MAPPING_BODY ...\n"
            data = obj_head
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
