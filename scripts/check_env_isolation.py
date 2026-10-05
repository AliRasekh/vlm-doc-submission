#!/usr/bin/env python3
"""Fail-fast launcher check + thin wrapper notes.

PYTHONNOUSERSITE must be set in the *process environment before* Python starts.
Setting it inside an already-running interpreter does not unload user-site modules.
"""

from __future__ import annotations

import os
import sys


def require_no_usersite() -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SystemExit(
            "PYTHONNOUSERSITE=1 must be exported by the shell/Slurm launcher "
            "before invoking Python (setting it inside main() is ineffective)."
        )


if __name__ == "__main__":
    require_no_usersite()
    print("ok: PYTHONNOUSERSITE=1")
