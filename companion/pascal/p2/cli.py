#!/usr/bin/env python3
"""P2 single-image/question CLI for InternVL3-1B (development/demo)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    os.chdir(REPO_ROOT)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT / "companion" / "pascal"))

    parser = argparse.ArgumentParser(
        description="Pascal P2 InternVL document QA (single image + question)"
    )
    parser.add_argument("--image", default=None, help="Path to document image")
    parser.add_argument("--question", default=None, help="Question text")
    parser.add_argument(
        "--max-num",
        type=int,
        default=12,
        help="Dynamic tile cap: 12 (default), 4 or 1 (reduced visual budgets)",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Use a development preset id (see --list-presets)",
    )
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Load model and print runtime details, then exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print exactly one JSON document on stdout",
    )
    args = parser.parse_args()

    # Import helpers first (no model load).
    from p2.cli_json import emit_json, stdout_to_stderr
    from p2.presets import get_preset, list_presets
    from p2.validate import ValidationError

    if args.list_presets:
        # Pure JSON list on stdout (no model).
        emit_json(list_presets())
        return 0

    # For --json: keep ALL load/inference notices on stderr, including import-time.
    # Engine import is deferred into the redirected section when --json.
    def _run_inference() -> dict:
        from p2.engine import get_engine

        engine = get_engine()
        meta = engine.ensure_loaded()
        if args.load_only:
            return meta

        image = args.image
        question = args.question
        max_num = args.max_num
        max_num_explicit = any(
            a == "--max-num" or a.startswith("--max-num=") for a in sys.argv[1:]
        )
        if args.preset:
            p = get_preset(args.preset)
            image = p["image_path"]
            question = p["question"]
            if not max_num_explicit and p.get("default_cap") is not None:
                max_num = int(p["default_cap"])
            print(
                f"preset={p['id']} qid={p['question_id']} "
                f"(references withheld from model)",
                file=sys.stderr,
            )
        if not image or not question:
            raise ValidationError("provide --image/--question or --preset")
        return engine.predict(
            image_path=image, question=question, max_num=max_num
        )

    try:
        if args.json or args.load_only:
            with stdout_to_stderr():
                result = _run_inference()
            emit_json(result)
            return 0

        # Human-readable mode: notices may appear on stderr; prediction on stdout.
        with stdout_to_stderr():
            from p2.engine import get_engine

            engine = get_engine()
            engine.ensure_loaded()
            image = args.image
            question = args.question
            max_num = args.max_num
            max_num_explicit = any(
                a == "--max-num" or a.startswith("--max-num=") for a in sys.argv[1:]
            )
            if args.preset:
                p = get_preset(args.preset)
                image = p["image_path"]
                question = p["question"]
                if not max_num_explicit and p.get("default_cap") is not None:
                    max_num = int(p["default_cap"])
                print(
                    f"preset={p['id']} qid={p['question_id']} "
                    f"(references withheld from model)",
                    file=sys.stderr,
                )
            if not image or not question:
                raise ValidationError("provide --image/--question or --preset")
            result = engine.predict(
                image_path=image, question=question, max_num=max_num
            )
        print(result["prediction"])
        print(
            f"[cap={result['max_num_cap']} tiles={result['actual_tile_count']} "
            f"request_s={result['request_seconds']:.3f} "
            f"scope={result['request_timing_scope']} "
            f"dtype={result['dtype']} backend={result['attention_backend_label']}]",
            file=sys.stderr,
        )
        return 0
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
