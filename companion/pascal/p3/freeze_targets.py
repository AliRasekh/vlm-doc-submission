#!/usr/bin/env python3
"""Freeze QID→target_answer mapping for P3 (identical across seeds)."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    man_path = REPO / "data/manifests/docvqa_train_subset_v1.json"
    ans_path = REPO / "data/cache/docvqa_train_subset_v1_answers.json"
    out = REPO / "companion/pascal/artifacts/p3/frozen_qid_target_map.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    man = json.loads(man_path.read_text())
    ans = json.loads(ans_path.read_text())
    from_answers: dict[int, str] = {}
    for r in ans.get("answers") or []:
        qid = int(r["question_id"])
        if r.get("target_answer") and str(r["target_answer"]).strip():
            from_answers[qid] = str(r["target_answer"]).strip()
        else:
            for a in r.get("answers") or []:
                if str(a).strip():
                    from_answers[qid] = str(a).strip()
                    break

    mapping: dict[str, str] = {}
    mismatches = []
    for e in man["examples"]:
        qid = int(e["question_id"])
        if e.get("target_answer") and str(e["target_answer"]).strip():
            t = str(e["target_answer"]).strip()
        elif qid in from_answers:
            t = from_answers[qid]
        else:
            raise KeyError(f"missing target for qid={qid}")
        if qid in from_answers and from_answers[qid] != t:
            mismatches.append({"qid": qid, "manifest": t, "answers": from_answers[qid]})
        mapping[str(qid)] = t

    # Canonical digest over sorted qid→target pairs
    canon = json.dumps(mapping, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(canon.encode()).hexdigest()
    doc = {
        "task": "pascal_p3_frozen_qid_target_map",
        "n": len(mapping),
        "target_selection_rule": "first_nonempty_source_listed_answer",
        "manifest": str(man_path.as_posix()),
        "manifest_sha256": sha256_file(man_path),
        "answers": str(ans_path.as_posix()),
        "answers_sha256": sha256_file(ans_path),
        "mapping_sha256": digest,
        "rng_independent": True,
        "note": (
            "Original trainer resolves targets from manifest target_answer / answers "
            "sidecar without RNG. This file freezes that identity for P3 provenance."
        ),
        "mismatches_manifest_vs_answers": mismatches,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mapping": mapping,
    }
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"out": str(out), "n": len(mapping), "mapping_sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
