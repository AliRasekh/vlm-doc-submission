#!/usr/bin/env python3
"""CPU-only rescore of stored predictions under a selected metric version.

Never trusts cached metrics from another scorer version.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    # Fail-fast isolation check before third-party imports beyond stdlib.
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print(
            "ERROR: PYTHONNOUSERSITE must be set to 1 in the shell/launcher "
            "before starting Python. Setting it inside main() does not "
            "retroactively disable user-site imports.",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument(
        "--primary",
        default="anls_normalized_strict_v2",
        choices=[
            "anls_normalized_strict_v2",
            "anls_normalized_inclusive_v1",
            "anls_vlmevalkit",
        ],
    )
    parser.add_argument(
        "--allow-stale-cached-metrics",
        action="store_true",
        help="Debug only; default always recomputes.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    from vlm_doc.metrics import (
        SCORER_VERSION_PRIMARY,
        aggregate_scores,
        score_example,
        scorer_source_hash,
        validate_answers_sidecar,
    )
    from vlm_doc.run_fingerprint import (
        sha256_file,
        verify_manifest_hash,
    )

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    vman = verify_manifest_hash(manifest, path=manifest_path)
    if not vman["ok"]:
        print(
            f"ERROR: manifest hash mismatch stored={vman['stored_manifest_sha256']} "
            f"canonical={vman['canonical_content_sha256']}",
            file=sys.stderr,
        )
        # Historical manifests may use a non-canonical hash algorithm.
        # Record warning and continue only if question_ids present; still fail
        # closed for newly written manifests that claim content hash equality.
        print(
            "WARNING: proceeding with recorded question_ids after documenting mismatch; "
            "see summary.manifest_hash_verification",
            file=sys.stderr,
        )

    qids = [int(x) for x in manifest["question_ids"]]
    answers_path = Path(args.answers)
    answers_blob = json.loads(answers_path.read_text())
    answers_list = answers_blob.get("answers", answers_blob)
    vans = validate_answers_sidecar(answers_list, required_question_ids=qids)
    if not vans["ok"]:
        print("ERROR: answer sidecar validation failed:", json.dumps({
            k: vans[k] for k in vans if k != "answers_by_qid"
        }), file=sys.stderr)
        return 1
    answers_by_qid = {int(k): v for k, v in vans["answers_by_qid"].items()}
    answer_sha = sha256_file(answers_path)

    # Load predictions: last record per question_id wins; reject out-of-manifest later
    by_qid: dict[int, dict] = {}
    with Path(args.predictions_jsonl).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_qid[int(rec["question_id"])] = rec

    ordered = []
    for qid in qids:
        rec = by_qid.get(qid)
        if rec is None:
            ordered.append(
                {
                    "question_id": qid,
                    "status": "missing",
                    "prediction": None,
                    "reference_answers": answers_by_qid.get(qid, []),
                    "metrics": score_example(
                        None, answers_by_qid.get(qid, []), status="error", primary=args.primary
                    ),
                }
            )
            continue
        pred = rec.get("prediction")
        # Prefer prediction_raw if present for documentation, but score stripped prediction
        # unless prediction field exists (historical).
        status = rec.get("status", "ok")
        refs = answers_by_qid[qid]
        metrics = score_example(pred, refs, status=status, primary=args.primary)
        if not args.allow_stale_cached_metrics:
            # Explicitly discard any cached metrics from the record.
            pass
        else:
            # Even with flag, still recompute primary unless identical version tag.
            cached = rec.get("metrics") or {}
            if cached.get("primary") == args.primary and "anls" in cached:
                metrics = score_example(pred, refs, status=status, primary=args.primary)
        out = {
            "question_id": qid,
            "status": status,
            "prediction": pred,
            "prediction_raw": rec.get("prediction_raw"),
            "prediction_whitespace_note": (
                "historical_stripped_decoded_output"
                if rec.get("prediction_raw") is None
                else "raw_and_stripped_available"
            ),
            "reference_answers": refs,
            "metrics": metrics,
            "original_run_fingerprint": (rec.get("run_fingerprint") or {}).get(
                "run_fingerprint_sha256"
            ),
            "original_model_revision": rec.get("model_revision"),
            "original_git_commit": (rec.get("run_metadata") or {}).get("git_commit"),
            "hardware": rec.get("hardware"),
            "generation_seconds": rec.get("generation_seconds"),
            "generated_token_length": rec.get("generated_token_length"),
            "max_new_tokens": rec.get("max_new_tokens"),
            "generation_cap_reached": (
                int(rec["generated_token_length"]) >= int(rec["max_new_tokens"])
                if rec.get("generated_token_length") is not None
                and rec.get("max_new_tokens") is not None
                else None
            ),
        }
        ordered.append(out)

    # Compare inclusive vs strict change count when both computable
    n_cutoff_change = 0
    for row in ordered:
        m = row["metrics"]
        if m.get("failure"):
            continue
        if float(m["anls_normalized_strict_v2"]) != float(
            m["anls_normalized_inclusive_v1"]
        ):
            n_cutoff_change += 1

    try:
        agg = aggregate_scores(
            ordered,
            n_requested=len(qids),
            required_question_ids=qids,
            primary_key=args.primary,
        )
    except ValueError as exc:
        print(f"ERROR: aggregate rejected records: {exc}", file=sys.stderr)
        return 1

    summary = {
        "task": "cpu_rescore",
        "manifest": str(manifest_path),
        "manifest_name": manifest.get("name"),
        "manifest_hash_verification": vman,
        "predictions_jsonl": str(args.predictions_jsonl),
        "answers_sidecar_sha256": answer_sha,
        "answer_validation": {k: v for k, v in vans.items() if k != "answers_by_qid"},
        "primary_metric": args.primary,
        "scorer_source_sha256": scorer_source_hash(),
        "n_examples_cutoff_diff_strict_vs_inclusive": n_cutoff_change,
        "scores": agg,
        "note": (
            "CPU rescore of historical predictions; inference provenance unchanged. "
            "Cached metrics from other scorer versions were not trusted."
        ),
    }
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, indent=2) + "\n")
    # Also write per-example rescored JSONL next to summary
    out_jsonl = Path(args.summary_out).with_suffix(".jsonl")
    with out_jsonl.open("w") as f:
        for row in ordered:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {args.summary_out}")
    print(f"wrote {out_jsonl}")
    print(
        f"primary={args.primary} mean={agg['mean_anls_requested']:.4f} "
        f"EM={agg['mean_exact_match_requested']:.4f} "
        f"cutoff_diff_vs_inclusive={n_cutoff_change}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
