#!/usr/bin/env python3
"""Deterministic illustrative gallery for visual-input dependence (no refs in model inputs).

Presentation-only rendering: does not modify experimental input images on disk.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image

from .paths import REPO_ROOT, VD_ART, VD_RESULTS, condition_paths

WHITE_NOTE = "Intentionally blank input; same original dimensions as the correct document."

CONDITION_HEADINGS = {
    "clean": "A. Correct document",
    "white": "B. Blank white input",
    "unrelated": "C. Unrelated document",
}


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


def _thumb_for_display(im: Image.Image, *, max_w: int, max_h: int) -> Image.Image:
    """Scale uniformly to fit box; never stretch aspect ratio."""
    scale = min(max_w / max(im.width, 1), max_h / max(im.height, 1))
    tw = max(1, int(round(im.width * scale)))
    th = max(1, int(round(im.height * scale)))
    return im.resize((tw, th), Image.Resampling.LANCZOS)


def render_gallery_figure(
    *,
    qid: int,
    question: str,
    panels: list[tuple[str, Image.Image, str | None]],
    out_pdf: Path,
    out_png: Path,
) -> None:
    """Compose one illustrative figure with vector text + raster thumbnails."""
    clean_im = panels[0][1]
    white_im = panels[1][1]
    unrelated_im = panels[2][1]
    shared_max_w, shared_max_h = 560, 300
    clean_thumb = _thumb_for_display(clean_im, max_w=shared_max_w, max_h=shared_max_h)
    white_thumb = white_im.resize(clean_thumb.size, Image.Resampling.NEAREST)
    unrelated_thumb = _thumb_for_display(
        unrelated_im, max_w=shared_max_w, max_h=shared_max_h
    )
    display_panels = [
        ("clean", clean_thumb, clean_im.size, panels[0][2]),
        ("white", white_thumb, white_im.size, panels[1][2]),
        ("unrelated", unrelated_thumb, unrelated_im.size, panels[2][2]),
    ]

    q_lines = textwrap.wrap(question.strip() or "(no question)", width=88)
    # Height budget: header + each panel (heading + image + captions)
    img_heights = [t.height for _, t, _, _ in display_panels]
    fig_h = 1.15 + 0.28 * len(q_lines)
    for key, thumb, _native, pred in display_panels:
        n_pred = max(1, len(textwrap.wrap(pred or "(empty prediction)", width=86)))
        fig_h += 0.38 + thumb.height / 95.0 + 0.22 * n_pred
        if key == "white":
            fig_h += 0.28
    fig_w = 8.4
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)
    fig.patch.set_facecolor("white")

    # Use a gridspec with one column; varying row heights.
    height_ratios: list[float] = [0.55 + 0.22 * len(q_lines)]
    for key, thumb, _native, pred in display_panels:
        n_pred = max(1, len(textwrap.wrap(pred or "(empty prediction)", width=86)))
        # heading + image + size + optional white note + prediction label/lines
        text_h = 0.55 + 0.28 * n_pred + (0.35 if key == "white" else 0.0)
        height_ratios.append(thumb.height / 95.0 + text_h)

    gs = fig.add_gridspec(
        nrows=1 + len(display_panels),
        ncols=1,
        height_ratios=height_ratios,
        hspace=0.18,
        left=0.06,
        right=0.97,
        top=0.97,
        bottom=0.03,
    )

    ax_h = fig.add_subplot(gs[0, 0])
    ax_h.set_axis_off()
    ax_h.set_xlim(0, 1)
    ax_h.set_ylim(0, 1)
    ax_h.text(
        0.0,
        0.95,
        f"QID {qid}",
        fontsize=12,
        fontweight="bold",
        fontname="DejaVu Sans",
        va="top",
        transform=ax_h.transAxes,
    )
    y = 0.70
    for line in q_lines:
        ax_h.text(
            0.0,
            y,
            line,
            fontsize=10.5,
            fontname="DejaVu Sans",
            va="top",
            transform=ax_h.transAxes,
        )
        y -= 0.28

    for i, (key, thumb, native_wh, pred) in enumerate(display_panels):
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.set_axis_off()
        # Nested layout: left image, right reserved unused; text above/below via fig.text is harder
        # Draw heading + image + captions inside this axes using blended coordinates.
        nw, nh = int(native_wh[0]), int(native_wh[1])
        pred_lines = textwrap.wrap(pred or "(empty prediction)", width=86) or [""]
        heading = CONDITION_HEADINGS[key]

        # Child axes for the thumbnail only, leaving room for text above/below.
        # Fraction of this subplot for the image.
        img_frac = (thumb.height / 95.0) / height_ratios[i + 1]
        img_frac = min(0.62, max(0.28, img_frac))
        # heading at top of subplot
        ax.text(
            0.0,
            1.0,
            heading,
            fontsize=10.5,
            fontweight="bold",
            fontname="DejaVu Sans",
            va="top",
            transform=ax.transAxes,
        )
        # image inset
        inset = ax.inset_axes([0.0, 1.0 - 0.12 - img_frac, thumb.width / (fig_w * 95.0), img_frac])
        inset.imshow(thumb, aspect="equal")
        inset.set_xticks([])
        inset.set_yticks([])
        if key == "white":
            for spine in inset.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#888888")
                spine.set_linewidth(1.3)
            inset.set_frame_on(True)
        else:
            for spine in inset.spines.values():
                spine.set_visible(False)

        text_y = 1.0 - 0.14 - img_frac - 0.02
        ax.text(
            0.0,
            text_y,
            f"Original input size: {nw}×{nh} px (PDF thumbnail is display-only).",
            fontsize=8.5,
            fontname="DejaVu Sans",
            va="top",
            color="#333333",
            transform=ax.transAxes,
        )
        text_y -= 0.10
        if key == "white":
            ax.text(
                0.0,
                text_y,
                WHITE_NOTE,
                fontsize=8.5,
                fontstyle="italic",
                fontname="DejaVu Sans",
                va="top",
                color="#333333",
                transform=ax.transAxes,
            )
            text_y -= 0.10
        ax.text(
            0.0,
            text_y,
            "Model prediction:",
            fontsize=8.5,
            fontweight="bold",
            fontname="DejaVu Sans",
            va="top",
            transform=ax.transAxes,
        )
        text_y -= 0.09
        for line in pred_lines:
            ax.text(
                0.0,
                text_y,
                line,
                fontsize=9.0,
                fontname="DejaVu Sans",
                va="top",
                transform=ax.transAxes,
            )
            text_y -= 0.085

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_png, format="png", dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def build_gallery() -> dict[str, Any]:
    clean = _load(condition_paths("clean")["jsonl"])
    white = _load(condition_paths("white")["jsonl"])
    unrelated = _load(condition_paths("unrelated")["jsonl"])
    mapping = json.loads((VD_ART / "unrelated_mapping.json").read_text())["mapping"]
    selected = select_qids(clean, white, unrelated, k=4)

    out_dir = VD_RESULTS / "visual_dependence_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("gallery_*.png"):
        p.unlink()
    for p in out_dir.glob("gallery_*.pdf"):
        p.unlink()

    entries = []
    for i, sel in enumerate(selected):
        qid = sel["question_id"]
        c, w, u = clean[qid], white[qid], unrelated[qid]
        m = mapping[str(qid)]
        clean_img = Image.open(REPO_ROOT / c["original_image_relpath"]).convert("RGB")
        ww, hh = c["original_size_wh"]
        # Presentation-only white canvas (do not write into data/images/).
        white_img = Image.new("RGB", (int(ww), int(hh)), (255, 255, 255))
        donor = Image.open(REPO_ROOT / m["donor_image_relpath"]).convert("RGB")
        panels = [
            ("clean", clean_img, c.get("prediction")),
            ("white", white_img, w.get("prediction")),
            ("unrelated", donor, u.get("prediction")),
        ]
        out_pdf = out_dir / f"gallery_{i:02d}_qid_{qid}.pdf"
        out_png = out_dir / f"gallery_{i:02d}_qid_{qid}.png"
        render_gallery_figure(
            qid=qid,
            question=str(c.get("question") or ""),
            panels=panels,
            out_pdf=out_pdf,
            out_png=out_png,
        )
        for im in (clean_img, white_img, donor):
            im.close()

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
                "figure_pdf": str(out_pdf.relative_to(REPO_ROOT)),
                "label": "illustrative_not_representative",
                "presentation_note": (
                    "Labels/borders are presentation-only; experimental input "
                    "files under data/images/ were not modified."
                ),
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
