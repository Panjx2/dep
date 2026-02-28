from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_summary(path: Path) -> dict[str, float]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {r["perturbation_type"]: float(r["schema_compliance"]) for r in rows}


def _rect(x: float, y: float, w: float, h: float, fill: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" />'


def make_schema_chart(vanilla_summary: Path, repair_summary: Path, out_path: Path) -> None:
    van = read_summary(vanilla_summary)
    rep = read_summary(repair_summary)
    labels = sorted(set(van) | set(rep))

    width, height = 1400, 700
    margin_l, margin_r, margin_t, margin_b = 90, 40, 60, 190
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    n = len(labels)
    group_w = plot_w / max(1, n)
    bar_w = group_w * 0.35

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-size:22px;font-weight:bold}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.legend{font-size:14px}</style>',
        f'<text class="title" x="{width/2}" y="34" text-anchor="middle">Schema Compliance by Perturbation and Mode</text>',
    ]

    x0, y0 = margin_l, margin_t
    parts.append(f'<line class="axis" x1="{x0}" y1="{y0+plot_h}" x2="{x0+plot_w}" y2="{y0+plot_h}"/>')
    parts.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+plot_h}"/>')

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = y0 + plot_h * (1 - t)
        parts.append(f'<line class="grid" x1="{x0}" y1="{yy:.1f}" x2="{x0+plot_w}" y2="{yy:.1f}"/>')
        parts.append(f'<text x="{x0-10}" y="{yy+4:.1f}" text-anchor="end">{t:.2f}</text>')

    for i, label in enumerate(labels):
        gx = x0 + i * group_w + group_w * 0.15
        vv = van.get(label, 0.0)
        rv = rep.get(label, 0.0)

        vh = plot_h * vv
        rh = plot_h * rv
        parts.append(_rect(gx, y0 + plot_h - vh, bar_w, vh, "#4C78A8"))
        parts.append(_rect(gx + bar_w + group_w * 0.08, y0 + plot_h - rh, bar_w, rh, "#F58518"))

        tx = x0 + i * group_w + group_w * 0.5
        parts.append(f'<text x="{tx:.1f}" y="{y0+plot_h+16}" text-anchor="end" transform="rotate(-35 {tx:.1f},{y0+plot_h+16})">{label}</text>')

    lx = width - 230
    ly = 90
    parts.append(_rect(lx, ly, 18, 18, "#4C78A8"))
    parts.append(f'<text class="legend" x="{lx+26}" y="{ly+14}">vanilla</text>')
    parts.append(_rect(lx, ly + 30, 18, 18, "#F58518"))
    parts.append(f'<text class="legend" x="{lx+26}" y="{ly+44}">repair</text>')

    parts.append(f'<text x="24" y="{y0+plot_h/2}" transform="rotate(-90 24,{y0+plot_h/2})">schema_ok rate</text>')
    parts.append('</svg>')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create summary visualizations from evaluation outputs")
    parser.add_argument("--vanilla-summary", default="results/vanilla/summary.csv")
    parser.add_argument("--repair-summary", default="results/repair/summary.csv")
    parser.add_argument("--out", default="results/schema_ok_by_perturbation_mode.svg")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_schema_chart(Path(args.vanilla_summary), Path(args.repair_summary), Path(args.out))
