#!/usr/bin/env python3
"""Deterministic illustrative gallery for visual-input dependence (no refs in model inputs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .paths import REPO_ROOT, VD_ART, VD_RESULTS, condition_paths


def _load(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[int(r["question_id"])] = r
    return out


def _em(row: dict[str, Any]) -> float:
    return float((row.get("metrics") or {}).get("exact_match") or 0.0)


def select_qids(clean: dict, white: dict, unrelated: dict, k: int = 4) -> list[dict[str, Any]]:
    """Deterministic illustrative selection (not a representative sample).

    Prefer ascending QIDs where clean EM=1 and white EM=0.
    Prefer cases where unrelated prediction differs from clean.
    """
    cands = []
    for qid in sorted(clean):
        c, w, u = clean[qid], white.get(qid), unrelated.get(qid)
        if not (c and w and u):
            continue
        if _em(c) != 1.0:
            continue
        if _em(w) != 0.0:
            continue
        differs = (c.get("prediction") or "") != (u.get("prediction") or "")
        cands.append(
            {
                "question_id": qid,
                "rank_key": (0 if differs else 1, qid),
                "rationale": (
                    "clean EM=1, white EM=0"
                    + ("; unrelated prediction differs from clean" if differs else "")
                ),
            }
        )
    cands.sort(key=lambda x: x["rank_key"])
    if len(cands) < k:
        # Fallback: clean EM=1 only
        for qid in sorted(clean):
            if any(c["question_id"] == qid for c in cands):
                continue
            if _em(clean[qid]) == 1.0:
                cands.append(
                    {
                        "question_id": qid,
                        "rank_key": (2, qid),
                        "rationale": "fallback: clean EM=1 (illustrative)",
                    }
                )
            if len(cands) >= k:
                break
    return cands[:k]


def build_gallery() -> dict[str, Any]:
    clean = _load(condition_paths("clean")["jsonl"])
    white = _load(condition_paths("white")["jsonl"])
    unrelated = _load(condition_paths("unrelated")["jsonl"])
    mapping = json.loads((VD_ART / "unrelated_mapping.json").read_text())["mapping"]
    selected = select_qids(clean, white, unrelated, k=4)

    out_dir = VD_RESULTS / "visual_dependence_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear prior gallery images only inside this directory
    for p in out_dir.glob("*.png"):
        p.unlink()

    entries = []
    for i, sel in enumerate(selected):
        qid = sel["question_id"]
        c, w, u = clean[qid], white[qid], unrelated[qid]
        m = mapping[str(qid)]
        panels = []
        # clean original
        clean_img = Image.open(REPO_ROOT / c["original_image_relpath"]).convert("RGB")
        panels.append(("clean", clean_img, c.get("prediction")))
        # white synthetic
        ww, hh = c["original_size_wh"]
        white_img = Image.new("RGB", (int(ww), int(hh)), (255, 255, 255))
        panels.append(("white", white_img, w.get("prediction")))
        # unrelated donor (native donor geometry — do not resize to query)
        donor = Image.open(REPO_ROOT / m["donor_image_relpath"]).convert("RGB")
        panels.append(("unrelated donor", donor, u.get("prediction")))

        # Stack panels vertically so question + full predictions stay readable in the PDF.
        question = (c.get("question") or "").strip()
        target_w = 720
        pad = 12
        row_gap = 10
        header_lines = [
            f"qid={qid} — illustrative only, not a representative sample",
            f"Q: {question}",
        ]

        def _wrap(text: str, max_chars: int = 96) -> list[str]:
            words = (text or "").split()
            if not words:
                return [""]
            lines: list[str] = []
            cur = words[0]
            for wtok in words[1:]:
                trial = f"{cur} {wtok}"
                if len(trial) <= max_chars:
                    cur = trial
                else:
                    lines.append(cur)
                    cur = wtok
            lines.append(cur)
            return lines

        header_drawn = []
        for line in header_lines:
            header_drawn.extend(_wrap(line, 100))

        thumbs = []
        for label, im, pred in panels:
            # White panels are blank: keep them short so predictions stay readable.
            max_h = 140 if label.startswith("white") else 320
            scale = target_w / max(im.width, 1)
            th = max(1, int(im.height * scale))
            th = min(th, max_h)
            tw = max(1, int(im.width * (th / max(im.height, 1))))
            thumb = im.resize((tw, th))
            pred_lines = _wrap(f"{label} ({im.size[0]}x{im.size[1]})  pred: {pred or ''}", 100)
            thumbs.append((thumb, pred_lines))
            im.close()

        line_h = 16
        header_h = 8 + line_h * len(header_drawn)
        body_h = sum(t[0].height + 6 + line_h * len(t[1]) + row_gap for t in thumbs)
        total_w = target_w + 2 * pad
        total_h = header_h + body_h + pad
        canvas = Image.new("RGB", (total_w, total_h), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        y = 6
        for line in header_drawn:
            draw.text((pad, y), line, fill=(20, 20, 20))
            y += line_h
        y += 4
        for thumb, pred_lines in thumbs:
            canvas.paste(thumb, (pad + (target_w - thumb.width) // 2, y))
            y += thumb.height + 4
            for line in pred_lines:
                draw.text((pad, y), line, fill=(20, 20, 20))
                y += line_h
            y += row_gap
        out_png = out_dir / f"gallery_{i:02d}_qid_{qid}.png"
        canvas.save(out_png)

        entries.append(
            {
                "question_id": qid,
                "rationale": sel["rationale"],
                "question": c.get("question"),
                "clean_prediction": c.get("prediction"),
                "white_prediction": w.get("prediction"),
                "unrelated_prediction": u.get("prediction"),
                "donor_qid": m["donor_qid"],
                "query_size_wh": m["query_size_wh"],
                "donor_size_wh": m["donor_size_wh"],
                "figure": str(out_png.relative_to(REPO_ROOT)),
                "label": "illustrative_not_representative",
            }
        )

    meta = {
        "task": "pascal_visual_dependence_gallery",
        "selection_rule": (
            "Ascending QIDs with clean EM=1 and white EM=0, preferring unrelated "
            "prediction ≠ clean; fallback clean EM=1. Illustrative only."
        ),
        "entries": entries,
    }
    meta_path = VD_RESULTS / "visual_dependence_gallery.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"gallery": str(meta_path), "n": len(entries)}, indent=2))
    return meta


def main() -> int:
    build_gallery()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
