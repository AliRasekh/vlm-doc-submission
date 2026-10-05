#!/usr/bin/env python3
"""Export untracked P2.1 review packet (excludes credentials / private keys)."""

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
    parser.add_argument("--out", default="companion/pascal/artifacts/P2_1_review.txt")
    parser.add_argument("--bundle-path", default=None)
    parser.add_argument("--bundle-sha256", default=None)
    parser.add_argument("--tested-revision", default=None)
    args = parser.parse_args()
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    include = [
        "companion/pascal/README.md",
        "companion/pascal/.gitignore",
        "companion/pascal/requirements-ui.txt",
        "companion/pascal/ssh_github_local.md",
        "companion/pascal/p2/__init__.py",
        "companion/pascal/p2/runtime.py",
        "companion/pascal/p2/validate.py",
        "companion/pascal/p2/engine.py",
        "companion/pascal/p2/presets.py",
        "companion/pascal/p2/cli_json.py",
        "companion/pascal/p2/cli.py",
        "companion/pascal/p2/gradio_schema_patch.py",
        "companion/pascal/p2/app.py",
        "companion/pascal/p2/verify_job.py",
        "companion/pascal/p2/ui_browser_test.py",
        "companion/pascal/p2/export_review.py",
        "companion/pascal/scripts/p2_verify.sbatch",
        "companion/pascal/scripts/p2_demo.sbatch",
        "companion/pascal/tests/test_p2_validate.py",
        "companion/pascal/tests/test_p2_cli_json.py",
        "docs/companion/pascal/PLAN.md",
        "docs/companion/pascal/PROGRESS.md",
        "docs/companion/pascal/DECISIONS.md",
        "docs/companion/pascal/P1_RUNTIME_CLARIFICATION.md",
        "docs/companion/pascal/P1_RESULTS.md",
        "docs/companion/pascal/P2_DEMO.md",
        "docs/companion/pascal/P2_VERIFICATION.md",
        "companion/pascal/results/P1_RESULTS.md",
        "companion/pascal/results/p2_verification_summary.json",
        "companion/pascal/results/p2_1_verification_summary.json",
        "companion/pascal/artifacts/p2_1_verify/verification.json",
        "companion/pascal/artifacts/p2_1_verify/bf16_probe.json",
        "companion/pascal/artifacts/p2_1_verify/runtime_details.json",
        "companion/pascal/artifacts/p2_1_verify/cli_exact_success.json",
        "companion/pascal/artifacts/p2_1_verify/browser_ui.json",
    ]
    # Include screenshots if present (binary noted by hash only in inventory;
    # embed as path listing — skip raw binary dump)
    shot_dir = REPO / "companion/pascal/artifacts/p2_1_verify/screenshots"
    shot_rels = []
    if shot_dir.is_dir():
        for p in sorted(shot_dir.glob("*.png")):
            rel = str(p.relative_to(REPO))
            shot_rels.append(rel)
            include.append(rel)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# PASCAL P2.1 REVIEW PACKET",
        f"# export_timestamp_utc={ts}",
        f"# final_commit={git_sha()}",
        f"# tested_revision={args.tested_revision or git_sha()}",
        f"# repo_root={REPO}",
        "# EXCLUDED: credentials, private keys, uploads, model weights, manifests bulk.",
        "",
    ]
    if args.bundle_path:
        lines += [
            "## GIT_BUNDLE",
            f"path={args.bundle_path}",
            f"sha256={args.bundle_sha256 or 'UNKNOWN'}",
            "",
        ]
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
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            lines += [
                "=" * 80,
                f"BEGIN_BINARY path={rel} bytes={p.stat().st_size} sha256={sha256_file(p)}",
                f"END_BINARY path={rel}",
                "",
            ]
            continue
        data = p.read_text(errors="replace")
        # Redact accidental private key material
        if "BEGIN OPENSSH PRIVATE KEY" in data or "BEGIN RSA PRIVATE KEY" in data:
            lines += [
                "=" * 80,
                f"BEGIN_FILE path={rel} REDACTED_PRIVATE_KEY_MATERIAL",
                f"END_FILE path={rel}",
                "",
            ]
            continue
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
    digest = hashlib.sha256(text.encode()).hexdigest()
    print(f"Wrote {out} size={len(text.encode())} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
