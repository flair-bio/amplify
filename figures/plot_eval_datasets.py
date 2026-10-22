"""Evaluate module: the evaluation datasets, grouped by source and coloured by task type.

Mirrors the evaluation dataset overlays plus the ProteinGym runners.

    python figures/plot_eval_datasets.py --outdir figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from plot_data_pipeline import GREEN, GREY, LANE1_BG, LANE2_BG, new_axes, rbox, save
from plot_masked_sequence import MASK_BG, TEXT_COLOR

SUB = "#444444"                                # darker than GREY so small text stays crisp on slides
LOGO = Path(__file__).parent / "assets" / "hf-logo.png"

TASK_COLORS = {
    "classification": MASK_BG,
    "multi-label": "#C99BF5",
    "regression": GREEN,
    "per-residue": "#9AA8A0",
    "contacts": "#5B2E7A",
}

LANES = [
    ("Biomap research", "per protein · per residue · per pair", LANE1_BG, [
        ("Localization", "10 classes", "classification"),
        ("Solubility", "binary", "classification"),
        ("Stability", "designed proteins", "regression"),
        ("SSP-Q3", "helix / strand / coil", "per-residue"),
        ("Contact map", "pairs with |i−j| ≥ 6", "contacts"),
    ]),
    ("UniProtKB  ·  InterPro", "per protein", LANE1_BG, [
        ("Molecular function", "GO  ·  1,225 labels", "multi-label"),
        ("Biological process", "GO  ·  1,899 labels", "multi-label"),
        ("Cellular component", "GO  ·  557 labels", "multi-label"),
        ("ProtParam", "31 properties", "regression"),
        ("Family", "InterPro  ·  7,386 labels", "multi-label"),
        ("Domain", "InterPro  ·  7,426 labels", "multi-label"),
        ("Homol. superfamily", "InterPro  ·  1,515 labels", "multi-label"),
        ("Repeat", "InterPro  ·  150 labels", "multi-label"),
        ("Active site", "InterPro  ·  84 labels", "multi-label"),
        ("Binding site", "InterPro  ·  55 labels", "multi-label"),
        ("Conserved site", "InterPro  ·  477 labels", "multi-label"),
    ]),
    ("UniProtKB features", "per residue", LANE1_BG, [
        ("Secondary structure", "4 classes", "per-residue"),
        ("Topology", "13 classes", "per-residue"),
        ("Membrane pass", "3 classes", "per-residue"),
        ("Peptides", "signal · transit · pro", "per-residue"),
        ("Functional sites", "active · binding · DNA", "per-residue"),
        ("Structures", "repeat · coiled coil · ZnF", "per-residue"),
        ("Domains", "binary", "per-residue"),
        ("Regions", "binary", "per-residue"),
        ("PTMs", "5 classes", "per-residue"),
        ("Phosphorylation", "6 classes", "per-residue"),
        ("Glycosylation", "binary", "per-residue"),
        ("Lipidation", "18 classes", "per-residue"),
    ]),
]

EXTRA = [
    ("ProteinGym", ["variant effects", "DMS + clinical", "zero-shot +", "supervised"]),
]


def chip(ax, x, y, w, h, name, sub, kind):
    rbox(ax, x, y, w, h, "white", r=0.18, z=1)
    rbox(ax, x + 0.15, y + 0.2, 0.2, h - 0.4, TASK_COLORS[kind], r=0.08, z=2)
    ax.text(x + 0.55, y + h * 0.66, name, ha="left", va="center", fontsize=17, fontweight="bold", color=TEXT_COLOR)
    ax.text(x + 0.55, y + h * 0.3, sub, ha="left", va="center", fontsize=15.5, color=SUB)


def plot(out: Path, transparent: bool, dpi: int):
    WM, pad, gap, per_row = 33.0, 0.45, 0.25, 6
    cw = (WM - 2 * pad - (per_row - 1) * gap) / per_row
    chh, rgap, head, lgap = 1.75, 0.3, 1.05, 0.45
    SW = 5.6                                       # standalone column width
    sx = WM + lgap
    W = sx + SW

    def lane_h(n):
        rows = -(-n // per_row)
        return rows * chh + (rows - 1) * rgap + pad + head

    heights = [lane_h(len(items)) for *_, items in LANES]
    total = sum(heights) + lgap * (len(heights) - 1)
    top = total + 1.4
    fig, ax = new_axes(-0.2, W + 0.2, -0.2, top + 0.2)

    # Header: Hub + legend.
    hy = top - 0.6
    ax.add_artist(AnnotationBbox(OffsetImage(plt.imread(LOGO), zoom=50 / 1024), (0.55, hy), frameon=False))
    ax.text(1.25, hy, "32 dataset configs, all on the Hugging Face Hub", ha="left", va="center", fontsize=22,
            fontweight="bold", color=TEXT_COLOR)
    lx = W
    for kind in reversed(list(TASK_COLORS)):
        tw = 0.215 * len(kind) + 0.6
        lx -= tw + 0.5
        rbox(ax, lx, hy - 0.25, 0.22, 0.5, TASK_COLORS[kind], r=0.06, z=2)
        ax.text(lx + 0.4, hy, kind, ha="left", va="center", fontsize=17, color=TEXT_COLOR)

    y = top - 1.4 + lgap
    lane_ys = []
    for (title, subtitle, bg, items), h in zip(LANES, heights):
        y -= h + lgap
        lane_ys.append(y)
        rbox(ax, 0, y, WM, h, bg, r=0.3, z=0)
        ax.text(pad - 0.05, y + h - 0.5, title, ha="left", va="center", fontsize=21, fontweight="bold",
                color=TEXT_COLOR)
        ax.text(WM - pad + 0.05, y + h - 0.5, subtitle, ha="right", va="center", fontsize=17, color=SUB,
                style="italic")
        for k, (name, sub, kind) in enumerate(items):
            r, c = divmod(k, per_row)
            cx = pad + c * (cw + gap)
            cy = y + h - head - (r + 1) * chh - r * rgap
            chip(ax, cx, cy, cw, chh, name, sub, kind)

    # Zero-shot note in the free slot of the first lane.
    nx = pad + 5 * (cw + gap)
    ny = lane_ys[0] + heights[0] - head - chh
    ax.add_patch(FancyBboxPatch((nx, ny), cw, chh, boxstyle="round,pad=0,rounding_size=0.18", fc="none",
                                ec=GREEN, lw=2.2, ls=(0, (4, 3)), zorder=1))
    ax.text(nx + 0.3, ny + chh * 0.66, "Pseudo-perplexity", ha="left", va="center", fontsize=17,
            fontweight="bold", color=TEXT_COLOR)
    ax.text(nx + 0.3, ny + chh * 0.3, "zero-shot, any dataset", ha="left", va="center", fontsize=15.5,
            color=SUB)

    # Standalone benchmarks: vertical column on the right.
    sy, sh = lane_ys[-1], top - 1.4 - lane_ys[-1]
    rbox(ax, sx, sy, SW, sh, LANE2_BG, r=0.3, z=0)
    ax.text(sx + pad - 0.05, sy + sh - 0.5, "Standalone", ha="left", va="center", fontsize=21,
            fontweight="bold", color=TEXT_COLOR)
    ax.text(sx + pad - 0.05, sy + sh - 1.45, "own runner or\nsourcing script", ha="left", va="center",
            fontsize=17, color=SUB, style="italic", linespacing=1.3)
    inner_top = sy + sh - 2.25
    cgap = 0.45
    ch2 = (inner_top - (sy + pad) - cgap) / 2
    for k, (name, lines) in enumerate(EXTRA):
        cy = inner_top - (k + 1) * ch2 - k * cgap
        rbox(ax, sx + 0.35, cy, SW - 0.7, ch2, "white", r=0.18, z=1)
        block = 0.8 + 0.72 * len(lines)
        ty = cy + ch2 / 2 + block / 2
        ax.text(sx + 0.65, ty - 0.2, name, ha="left", va="center", fontsize=19, fontweight="bold",
                color=TEXT_COLOR)
        for j, line in enumerate(lines):
            ax.text(sx + 0.65, ty - 1.05 - j * 0.72, line, ha="left", va="center", fontsize=17,
                    color=SUB)

    save(fig, out, transparent, dpi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    p.add_argument("--opaque", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    a = p.parse_args()
    plot(a.outdir / "eval_datasets.png", not a.opaque, a.dpi)


if __name__ == "__main__":
    main()
