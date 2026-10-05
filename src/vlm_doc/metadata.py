"""Git and runtime metadata helpers."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def collect_run_metadata(repo_root: Path) -> dict[str, Any]:
    commit = _run_git(["rev-parse", "HEAD"], repo_root)
    dirty = _run_git(["status", "--porcelain"], repo_root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": bool(dirty) if dirty is not None else None,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cwd": str(Path.cwd()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "slurm_qos": os.environ.get("SLURM_JOB_QOS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
