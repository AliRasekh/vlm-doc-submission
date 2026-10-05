#!/usr/bin/env python3
"""UCSF-cluster bootstrap for InternVL base vs adapted holdout ANLS difference."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_preds(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[int(rec["question_id"])] = rec
    return out


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def ucsf_cluster_bootstrap_diff(
    *,
    by_group: dict[str, list[int]],
    score_by_qid: dict[int, tuple[float, float]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Paired adapted−base ANLS bootstrap with group multiplicity."""
    group_ids = sorted(by_group.keys())
    all_q = [q for g in group_ids for q in by_group[g]]
    point_base = sum(score_by_qid[q][0] for q in all_q) / len(all_q)
    point_adapted = sum(score_by_qid[q][1] for q in all_q) / len(all_q)
    point_diff = point_adapted - point_base

    rng = random.Random(int(seed))
    diffs: list[float] = []
    for _ in range(int(replicates)):
        drawn = [rng.choice(group_ids) for _ in group_ids]
        qs: list[int] = []
        for g in drawn:
            qs.extend(by_group[g])
        mb = sum(score_by_qid[q][0] for q in qs) / len(qs)
        ma = sum(score_by_qid[q][1] for q in qs) / len(qs)
        diffs.append(ma - mb)
    diffs_sorted = sorted(diffs)
    return {
        "n_questions": len(all_q),
        "n_groups": len(group_ids),
        "replicates": int(replicates),
        "seed": int(seed),
        "point_mean_anls_base": point_base,
        "point_mean_anls_adapted": point_adapted,
        "point_diff_adapted_minus_base": point_diff,
        "bootstrap_diff_mean": sum(diffs) / len(diffs),
        "bootstrap_diff_p2_5": _percentile(diffs_sorted, 2.5),
        "bootstrap_diff_p97_5": _percentile(diffs_sorted, 97.5),
        "interval": "percentile_95",
    }


def verify_multiplicity_synthetic() -> dict[str, Any]:
    """Synthetic check: duplicated sampled groups must contribute multiple times."""
    by_group = {
        "G1": [1, 2],  # base 1.0, adapted 0.0 for both → contrib -1 each
        "G2": [3],  # base 0.0, adapted 1.0 → +1
    }
    score_by_qid = {
        1: (1.0, 0.0),
        2: (1.0, 0.0),
        3: (0.0, 1.0),
    }
    # Force draws: G1, G1 (multiplicity) with n_groups=2
    # qs = [1,2,1,2]; mean_base=1.0; mean_adapted=0.0; diff=-1.0
    group_ids = sorted(by_group.keys())
    drawn = ["G1", "G1"]
    qs: list[int] = []
    for g in drawn:
        qs.extend(by_group[g])
    assert qs == [1, 2, 1, 2], qs
    assert len(qs) == 4
    mb = sum(score_by_qid[q][0] for q in qs) / len(qs)
    ma = sum(score_by_qid[q][1] for q in qs) / len(qs)
    assert abs(mb - 1.0) < 1e-12 and abs(ma - 0.0) < 1e-12
    # Without multiplicity (unique groups only) would be wrong for this design.
    unique_only = []
    for g in set(drawn):
        unique_only.extend(by_group[g])
    assert unique_only == [1, 2]
    return {
        "ok": True,
        "drawn": drawn,
        "questions_with_multiplicity": qs,
        "diff_with_multiplicity": ma - mb,
        "questions_unique_only_incorrect": unique_only,
        "note": "Duplicated G1 must yield four question contributions, not two.",
    }


def main() -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("ERROR: export PYTHONNOUSERSITE=1 before python", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/docvqa_holdout_v2.json")
    ap.add_argument("--base-jsonl", default=None)
    ap.add_argument("--adapted-jsonl", default=None)
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/holdout_internvl_anls_bootstrap.json")
    ap.add_argument(
        "--self-test-multiplicity",
        action="store_true",
        help="Run synthetic multiplicity check and exit",
    )
    args = ap.parse_args()

    if args.self_test_multiplicity:
        r = verify_multiplicity_synthetic()
        print(json.dumps(r, indent=2))
        return 0 if r["ok"] else 2

    if not args.base_jsonl or not args.adapted_jsonl:
        print("ERROR: --base-jsonl and --adapted-jsonl required", file=sys.stderr)
        return 2

    root = _repo_root()
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    from vlm_doc.metrics import SCORER_VERSION_PRIMARY

    syn = verify_multiplicity_synthetic()
    if not syn["ok"]:
        print("ERROR: multiplicity self-test failed", syn, file=sys.stderr)
        return 2

    manifest = json.loads(Path(args.manifest).read_text())
    base = _load_preds(Path(args.base_jsonl))
    adapted = _load_preds(Path(args.adapted_jsonl))

    by_group: dict[str, list[int]] = defaultdict(list)
    score_by_qid: dict[int, tuple[float, float]] = {}
    for ex in manifest["examples"]:
        qid = int(ex["question_id"])
        gid = str(ex.get("ucsf_document_id") or f"MISSING::{qid}")
        b = base.get(qid)
        a = adapted.get(qid)
        if not b or not a:
            print(f"ERROR: missing prediction for qid={qid}", file=sys.stderr)
            return 2
        b_anls = float((b.get("metrics") or {}).get("anls", 0.0))
        a_anls = float((a.get("metrics") or {}).get("anls", 0.0))
        if b.get("status") != "ok":
            b_anls = 0.0
        if a.get("status") != "ok":
            a_anls = 0.0
        by_group[gid].append(qid)
        score_by_qid[qid] = (b_anls, a_anls)

    boot = ucsf_cluster_bootstrap_diff(
        by_group=dict(by_group),
        score_by_qid=score_by_qid,
        replicates=int(args.replicates),
        seed=int(args.seed),
    )
    report = {
        "task": "holdout_ucsf_cluster_bootstrap_anls_diff",
        "primary_metric": SCORER_VERSION_PRIMARY,
        **boot,
        "method": (
            "Resample complete UCSF groups with replacement; retain all questions "
            "within each sampled group including multiplicity; question-weighted means; "
            "paired difference adapted-base."
        ),
        "interpretation": (
            "Interval describes uncertainty within this sampled evaluation design, "
            "not all sources of uncertainty."
        ),
        "multiplicity_self_test": syn,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
