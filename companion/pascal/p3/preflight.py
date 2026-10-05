#!/usr/bin/env python3
"""P3 preflight: compatibility gate + short timed training probe (discarded)."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

from .gate_wrap import run_gate
from .paths import P3_ART, seed_dirs
from .train_wrap import run_train


def main() -> int:
    P3_ART.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rc = run_gate()
    gate_s = time.perf_counter() - t0
    if rc != 0:
        print("PREFLIGHT_GATE_FAILED", flush=True)
        return rc

    # Fresh process timing: 4 optimizer updates (~32 presentations), then discard.
    probe_seed = 42
    probe_root = seed_dirs(probe_seed)["root"]
    bak = P3_ART / "preflight_discard"
    bak.mkdir(parents=True, exist_ok=True)
    if probe_root.exists():
        dest = bak / "seed_42_before_probe"
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(probe_root), str(dest))

    t1 = time.perf_counter()
    trc = run_train(probe_seed, max_updates=4)
    probe_s = time.perf_counter() - t1
    # Discard probe artifacts so main runs start clean
    if probe_root.exists():
        shutil.rmtree(probe_root, ignore_errors=True)

    per_update = probe_s / 4.0 if probe_s > 0 else None
    # Budget: 3 * 200 updates + gate already done + 4 evals (~208 * ~3s * 4)
    est_train = (per_update or 10.0) * 200 * 3
    est_eval = 208 * 3.5 * 4
    est_total_h = (est_train + est_eval) / 3600.0
    report = {
        "task": "pascal_p3_preflight",
        "gate_seconds": gate_s,
        "probe_updates": 4,
        "probe_seconds": probe_s,
        "seconds_per_update": per_update,
        "est_three_train_seconds": est_train,
        "est_four_eval_seconds": est_eval,
        "est_main_hours": est_total_h,
        "recommended_sbatch_time": "04:00:00" if est_total_h < 3.5 else "04:00:00",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
        "probe_discarded": True,
        "train_probe_rc": trc,
    }
    out = P3_ART / "preflight.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("PREFLIGHT", json.dumps(report, indent=2), flush=True)
    return 0 if trc == 0 else trc


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
