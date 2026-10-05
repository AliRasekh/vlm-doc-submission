"""Minimal SVG line/bar charts without matplotlib (keep env stack unchanged)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def write_line_chart_svg(
    path: Path,
    *,
    xs: Sequence[float],
    ys: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    width: int = 720,
    height: int = 360,
    pad: int = 48,
) -> None:
    if len(xs) != len(ys) or not xs:
        raise ValueError("xs/ys must be nonempty and equal length")
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1.0
    if ymax == ymin:
        ymax = ymin + 1.0
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad

    def sx(x: float) -> float:
        return pad + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return pad + (1.0 - (y - ymin) / (ymax - ymin)) * plot_h

    pts = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys))
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{width/2:.1f}" y="24" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="14">{title}</text>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333" stroke-width="1"/>
  <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333" stroke-width="1"/>
  <polyline fill="none" stroke="#1f4e79" stroke-width="1.5" points="{pts}"/>
  <text x="{width/2:.1f}" y="{height-12}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="12">{xlabel}</text>
  <text x="16" y="{height/2:.1f}" text-anchor="middle" transform="rotate(-90 16 {height/2:.1f})" font-family="DejaVu Sans,sans-serif" font-size="12">{ylabel}</text>
  <text x="{pad}" y="{pad-8}" font-family="DejaVu Sans,sans-serif" font-size="10" fill="#444">y=[{ymin:.4g},{ymax:.4g}]</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)


def write_grouped_bar_svg(
    path: Path,
    *,
    categories: Sequence[str],
    series: dict[str, Sequence[float]],
    title: str,
    ylabel: str,
    width: int = 820,
    height: int = 400,
    pad: int = 56,
) -> None:
    if not categories or not series:
        raise ValueError("categories/series required")
    n_cat = len(categories)
    n_ser = len(series)
    colors = ["#1f4e79", "#b85c38", "#2a7f62", "#6b5b95", "#c4a35a", "#4a4a4a"]
    all_vals = [v for vals in series.values() for v in vals]
    ymin = 0.0
    ymax = max(all_vals) if all_vals else 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad
    group_w = plot_w / n_cat
    bar_w = group_w / (n_ser + 1.5)

    def sy(y: float) -> float:
        return pad + (1.0 - (y - ymin) / (ymax - ymin)) * plot_h

    parts = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="14">{title}</text>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>',
    ]
    for si, (name, vals) in enumerate(series.items()):
        color = colors[si % len(colors)]
        for ci, val in enumerate(vals):
            x = pad + ci * group_w + (si + 0.75) * bar_w
            y = sy(val)
            h = (height - pad) - y
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{max(h,0):.2f}" fill="{color}"/>'
            )
        # legend
        lx = pad + si * 140
        parts.append(
            f'<rect x="{lx}" y="{height-28}" width="12" height="12" fill="{color}"/>'
            f'<text x="{lx+16}" y="{height-18}" font-family="DejaVu Sans,sans-serif" font-size="11">{name}</text>'
        )
    for ci, cat in enumerate(categories):
        cx = pad + (ci + 0.5) * group_w
        parts.append(
            f'<text x="{cx:.1f}" y="{height-pad+14}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="11">{cat}</text>'
        )
    parts.append(
        f'<text x="16" y="{height/2:.1f}" text-anchor="middle" transform="rotate(-90 16 {height/2:.1f})" font-family="DejaVu Sans,sans-serif" font-size="12">{ylabel}</text>'
    )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")
