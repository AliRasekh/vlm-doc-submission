#!/usr/bin/env python3
"""Build deterministic development robustness subset (Phase 3B).

Sample 16 UCSF groups uniformly without replacement from sorted development
group IDs (seed 42). Within each group, select up to 4 questions via a
deterministic seeded shuffle. Selection does not use predictions or scores.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("ERROR: export PYTHONNOUSERSITE=1 before python", file=sys.stderr)
        return 2
    p = argparse.ArgumentParser()
    p.add_argument("--dev-manifest", default="data/manifests/docvqa_dev_v1.json")
    p.add_argument("--out", default="data/manifests/docvqa_dev_robustness_v1.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-groups", type=int, default=16)
    p.add_argument("--max-per-group", type=int, default=4)
    args = p.parse_args()

    root = _repo_root()
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    from vlm_doc.run_fingerprint import canonical_manifest_content_hash, sha256_file

    parent = json.loads(Path(args.dev_manifest).read_text())
    by_group: dict[str, list[dict]] = defaultdict(list)
    for ex in parent["examples"]:
        gid = ex.get("ucsf_document_id")
        if not gid:
            raise RuntimeError(f"missing ucsf_document_id for qid={ex.get('question_id')}")
        by_group[str(gid)].append(ex)

    group_ids = sorted(by_group.keys())
    rng = random.Random(int(args.seed))
    if len(group_ids) < args.n_groups:
        raise RuntimeError(f"only {len(group_ids)} groups; need {args.n_groups}")
    sampled_groups = sorted(rng.sample(group_ids, args.n_groups))

    selected: list[dict] = []
    membership = []
    for gid in sampled_groups:
        members = list(by_group[gid])
        # Deterministic shuffle within group (seed derived from global seed + group id).
        local = random.Random(f"{args.seed}:{gid}")
        local.shuffle(members)
        take = members[: min(args.max_per_group, len(members))]
        take_sorted = sorted(take, key=lambda e: int(e["question_id"]))
        qids = [int(e["question_id"]) for e in take_sorted]
        membership.append(
            {
                "ucsf_document_id": gid,
                "n_available": len(members),
                "n_selected": len(take_sorted),
                "question_ids": qids,
            }
        )
        selected.extend(take_sorted)

    selected = sorted(selected, key=lambda e: int(e["question_id"]))
    manifest = {
        "name": "docvqa_dev_robustness_v1",
        "role": "dev_robustness_subset",
        "parent_manifest": str(Path(args.dev_manifest).as_posix()),
        "parent_manifest_raw_file_sha256": sha256_file(Path(args.dev_manifest)),
        "selection": {
            "seed": int(args.seed),
            "n_groups_requested": int(args.n_groups),
            "max_questions_per_group": int(args.max_per_group),
            "method": (
                "sample_16_ucsf_groups_uniform_wo_replacement_from_sorted_ids; "
                "within_group_seeded_shuffle_take_up_to_4; independent_of_predictions"
            ),
            "sampled_ucsf_document_ids": sampled_groups,
            "membership_by_group": membership,
        },
        "n_examples": len(selected),
        "n_groups": len(sampled_groups),
        "question_ids": [int(e["question_id"]) for e in selected],
        "examples": selected,
    }
    manifest["manifest_sha256"] = canonical_manifest_content_hash(manifest)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    out.write_text(text)
    print(
        json.dumps(
            {
                "out": str(out),
                "n_examples": manifest["n_examples"],
                "n_groups": manifest["n_groups"],
                "manifest_sha256": manifest["manifest_sha256"],
                "raw_file_sha256": sha256_file(out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
