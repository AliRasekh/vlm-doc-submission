#!/usr/bin/env python3
"""Count train examples affected by the v1 double end-of-turn supervision bug.

Uses text tokenization only (fixed synthetic image-token expansion); does not
materialize document images.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("ERROR: export PYTHONNOUSERSITE=1 before python", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest", default="data/manifests/docvqa_train_subset_v1.json"
    )
    ap.add_argument(
        "--answers", default="data/cache/docvqa_train_subset_v1_answers.json"
    )
    ap.add_argument("--out", default="results/supervised_termination_audit_v1.json")
    ap.add_argument("--instruction", default="Answer the question using a single word or short phrase.")
    ap.add_argument("--num-fake-img-tokens", type=int, default=256)
    args = ap.parse_args()

    root = _repo_root()
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    import torch
    from transformers import AutoTokenizer

    snap = (
        root
        / ".hf_cache/hub/models--OpenGVLab--InternVL3-1B/snapshots"
        / "4415a3b810e636d11dfa86b0e9ba40bb00535aa8"
    )
    sys.path.insert(0, str(snap))
    from conversation import get_conv_template

    from vlm_doc.train.collator import resolve_assistant_termination

    tok = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL3-1B",
        trust_remote_code=True,
        revision="4415a3b810e636d11dfa86b0e9ba40bb00535aa8",
        use_fast=False,
    )
    eos_id = int(tok.convert_tokens_to_ids("<|im_end|>"))
    nl_id = int(tok.encode("\n", add_special_tokens=False)[-1])

    manifest = json.loads(Path(args.manifest).read_text())
    answers_blob = json.loads(Path(args.answers).read_text())
    # Prefer manifest target_answer when present
    targets: dict[int, str] = {}
    for e in manifest["examples"]:
        qid = int(e["question_id"])
        if e.get("target_answer"):
            targets[qid] = str(e["target_answer"]).strip()
    if not targets:
        for a in answers_blob.get("answers", answers_blob):
            targets[int(a["question_id"])] = str(
                a.get("target_answer") or (a.get("answers") or [""])[0]
            ).strip()

    img = "<img>" + ("<IMG_CONTEXT>" * int(args.num_fake_img_tokens)) + "</img>"
    n = 0
    n_would_append_second_eos = 0
    n_ends_eos_newline = 0
    n_fixed_drops_trailing = 0
    examples_sample = []

    for e in manifest["examples"]:
        qid = int(e["question_id"])
        ans = targets.get(qid, "").strip()
        if not ans:
            continue
        n += 1
        text = f"{e['question'].strip()}\n{args.instruction}"
        qm = f"<image>\n{text}".replace("<image>", img, 1)
        tp = get_conv_template("internvl2_5")
        tp.system_message = "SYS"
        tp.append_message(tp.roles[0], qm)
        tp.append_message(tp.roles[1], None)
        tf = get_conv_template("internvl2_5")
        tf.system_message = "SYS"
        tf.append_message(tf.roles[0], qm)
        tf.append_message(tf.roles[1], ans)
        prompt_ids = tok(tp.get_prompt(), return_tensors="pt")["input_ids"][0]
        full_ids = tok(tf.get_prompt(), return_tensors="pt")["input_ids"][0]
        if not torch.equal(full_ids[: prompt_ids.numel()], prompt_ids):
            raise RuntimeError(f"prefix mismatch qid={qid}")

        ends_eos_nl = (
            full_ids.numel() >= 2
            and int(full_ids[-2]) == eos_id
            and int(full_ids[-1]) == nl_id
        )
        if ends_eos_nl:
            n_ends_eos_newline += 1
        # v1 buggy condition
        would_append = int(full_ids[-1]) != eos_id
        if would_append:
            n_would_append_second_eos += 1
        fixed = resolve_assistant_termination(
            full_ids, prompt_len=int(prompt_ids.numel()), eos_id=eos_id
        )
        if fixed.numel() < full_ids.numel() or would_append:
            n_fixed_drops_trailing += 1
        if qid in (46350, 21537, 12611) or len(examples_sample) < 3:
            examples_sample.append(
                {
                    "question_id": qid,
                    "answer": ans,
                    "raw_tail_tokens": tok.convert_ids_to_tokens(full_ids[-6:].tolist()),
                    "raw_tail_ids": full_ids[-6:].tolist(),
                    "would_v1_append_second_eos": would_append,
                    "fixed_tail_tokens": tok.convert_ids_to_tokens(fixed[-6:].tolist()),
                    "fixed_tail_ids": fixed[-6:].tolist(),
                    "n_raw": int(full_ids.numel()),
                    "n_fixed": int(fixed.numel()),
                }
            )

    report = {
        "task": "supervised_termination_audit_v1",
        "n_examples_scanned": n,
        "n_template_ends_with_eos_newline": n_ends_eos_newline,
        "n_v1_would_append_second_eos": n_would_append_second_eos,
        "n_changed_by_v2_termination_fix": n_fixed_drops_trailing,
        "fraction_v1_would_append_second_eos": (
            n_would_append_second_eos / n if n else None
        ),
        "eos_id": eos_id,
        "newline_token_id_probe": nl_id,
        "method": (
            "Text-only tokenization with fixed synthetic IMG_CONTEXT expansion; "
            "no document image materialization. Detects v1 rule "
            "if full_ids[-1] != eos_id then append eos."
        ),
        "examples_sample": examples_sample,
        "ok": n > 0,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "examples_sample"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
