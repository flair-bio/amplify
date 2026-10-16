"""Evaluate module: the 7 registered downstream tasks, grouped by output level.

Mirrors modules/evaluate/src/tasks/*.py (@register_task) and configs/datasets/*.yaml.

    python figures/plot_downstream_tasks.py --outdir figures
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

from plot_data_pipeline import GREEN, GREY, LANE1_BG, LANE2_BG, new_axes, rbox, save
from plot_masked_sequence import MASK_BG, RESIDUE_BG, TEXT_COLOR

CARD_BG = "white"
COIL = "#9AA8A0"
MONO = "DejaVu Sans Mono"

UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

C = 0.78   # mini residue cell
G = 0.1


def cells(ax, x, y, seq, fc=RESIDUE_BG, highlight=None, fs=15):
    """Row of residue cells; highlight = {index: (facecolor, text, textcolor)}."""
    highlight = highlight or {}
    for k, aa in enumerate(seq):
        fcol, txt, tcol = highlight.get(k, (fc, aa, TEXT_COLOR))
        w = C
        rbox(ax, x, y, w, C, fcol, r=0.1, z=3)
        ax.text(x + w / 2, y + C / 2, txt, ha="center", va="center", fontsize=fs if len(txt) == 1 else 8.5,
                fontweight="bold", color=tcol, family=MONO, zorder=4)
        x += w + G
    return x - G


def arrow(ax, x0, x1, y0, y1=None, color=TEXT_COLOR):
    ax.annotate("", xy=(x1, y0 if y1 is None else y1), xytext=(x0, y0), arrowprops=dict(
        arrowstyle="-|>,head_width=0.2,head_length=0.3", lw=1.8, color=color, shrinkA=0, shrinkB=0), zorder=4)


def chip(ax, x, y, text, fc, w=None, tc=TEXT_COLOR, fs=17):
    w = w or 0.32 * len(text) + 0.7
    rbox(ax, x, y, w, C, fc, r=0.2, z=3)
    ax.text(x + w / 2, y + C / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc, zorder=4)
    return x + w


def card(ax, x, y, w, h, title, config, caption):
    rbox(ax, x, y, w, h, CARD_BG, r=0.25, z=1)
    ax.text(x + 0.4, y + h - 0.55, title, ha="left", va="center", fontsize=19, fontweight="bold", color=TEXT_COLOR)
    ax.text(x + w - 0.4, y + h - 0.55, config, ha="right", va="center", fontsize=13, color=GREY, family=MONO)
    ax.text(x + 0.4, y + 0.5, caption, ha="left", va="center", fontsize=15, color=GREY, style="italic")


def lane(ax, x, y, w, h, bg, title, subtitle):
    rbox(ax, x, y, w, h + 0.85, bg, r=0.3, z=0)
    ax.text(x + 0.45, y + h + 0.4, title, ha="left", va="center", fontsize=22, fontweight="bold", color=TEXT_COLOR)
    ax.text(x + w - 0.45, y + h + 0.4, subtitle, ha="right", va="center", fontsize=15, color=GREY, style="italic")


def contact_map(ax, x, y, size, n=14, highlight_long=True):
    cell = size / n
    m = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if abs(i - j) <= 1:
                m[i, j] = 1                       # local backbone
    for k in range(4):                            # antiparallel hairpin
        m[2 + k, 9 - k] = m[9 - k, 2 + k] = 2
    for k in range(3):                            # long-range parallel pair
        m[1 + k, 11 + k] = m[11 + k, 1 + k] = 2
    rbox(ax, x, y, size, size, "#F3F5F4", r=0.05, z=2)
    for i in range(n):
        for j in range(n):
            if m[i, j]:
                fc = RESIDUE_BG if m[i, j] == 1 else MASK_BG
                ax.add_patch(Rectangle(
                    (x + j * cell, y + size - (i + 1) * cell), cell * 0.92, cell * 0.92, fc=fc, ec="none", zorder=3))
    ax.text(x - 0.3, y + size / 2, "residue i", ha="center", va="center", fontsize=13, color=GREY, rotation=90)


def plot(out: Path, transparent: bool, dpi: int):
    W = 33.0
    pad = 0.5
    pady = 0.3                                    # vertical padding inside lanes
    gap = 0.6
    ch1, ch = 3.0, 3.75                           # card heights
    lh1, lane_h = ch1 + pady, ch + pady
    y3 = 0.0
    y2 = y3 + lane_h + 0.85 + 0.3
    y1 = y2 + lane_h + 0.85 + 0.3
    fig, ax = new_axes(-0.2, W + 0.2, -0.2, y1 + lh1 + 1.05)

    lane(ax, 0, y1, W, lh1, LANE1_BG, "Per sequence", "probe or fine-tune  ·  pooled embedding → head")
    lane(ax, 0, y2, W, lane_h, LANE1_BG, "Per residue  ·  per residue pair",
         "probe or fine-tune  ·  per-residue embeddings → head")
    lane(ax, 0, y3, W, lane_h, LANE2_BG, "Zero-shot", "as-is  ·  no training, uses the MLM head")

    # ---------------------------------------------------------------- lane 1: one label per protein
    inner = W - 2 * pad
    w3 = (inner - 2 * gap) / 3
    cy = y1 + pady
    vy = cy + 1.05
    seq = "MKTAYI"
    xs = [pad + k * (w3 + gap) for k in range(3)]

    x = xs[0]
    card(ax, x, cy, w3, ch1, "Classification", "sequence_classification", "localization (10 compartments)  ·  metal binding")
    e = cells(ax, x + 0.4, vy, seq)
    arrow(ax, e + 0.25, e + 1.1, vy + C / 2)
    chip(ax, e + 1.3, vy, "nucleus", MASK_BG, tc="white", w=3.0)

    x = xs[1]
    card(ax, x, cy, w3, ch1, "Regression", "sequence_regression", "optimal temperature  ·  ProtParam")
    e = cells(ax, x + 0.4, vy, seq)
    arrow(ax, e + 0.25, e + 1.1, vy + C / 2)
    chip(ax, e + 1.3, vy, "62 °C", GREEN, w=3.0)

    x = xs[2]
    card(ax, x, cy, w3, ch1, "Multi-label", "sequence_multilabel_classification",
         "GO terms  ·  InterPro entries")
    e = cells(ax, x + 0.4, vy, seq)
    arrow(ax, e + 0.25, e + 1.1, vy + C / 2)
    tx = e + 1.3
    for k, (term, on) in enumerate([("ATP binding", True), ("kinase activity", True), ("DNA binding", False)]):
        ty = vy + C / 2 + 0.5 - k * 0.5
        ax.text(tx, ty, ("✓ " if on else "✗ ") + term, ha="left", va="center", fontsize=14,
                fontweight="bold" if on else "normal", color=MASK_BG if on else GREY)

    # ---------------------------------------------------------------- lane 2: per residue / pair
    w2 = (inner - gap) / 2
    cy = y2 + pady
    vy = cy + 1.95

    x = pad
    card(ax, x, cy, w2, ch, "Token classification", "token_classification",
         "SSP-Q3  ·  UniProt residue features (sites, PTMs, topology…)")
    seg = UBIQUITIN[13:27]                         # ubiquitin 14-27 (PDB 1UBQ): end of β2, loop, start of α1
    ss = "EEEECCCCCHHHHH"
    xe = cells(ax, x + 0.4, vy, seg)
    col = {"E": (MASK_BG, "white"), "H": (GREEN, TEXT_COLOR), "C": (COIL, "white")}
    for k, s_ in enumerate(ss):
        fc, tc = col[s_]
        rbox(ax, x + 0.4 + k * (C + G), vy - C - 0.12, C, C, fc, r=0.1, z=3)
        ax.text(x + 0.4 + k * (C + G) + C / 2, vy - 0.12 - C / 2, s_, ha="center", va="center", fontsize=15,
                fontweight="bold", color=tc, family=MONO, zorder=4)
    lx = xe + 0.5
    for k, (lab, s_) in enumerate([("strand", "E"), ("helix", "H"), ("coil", "C")]):
        ly = vy + C - 0.15 - k * 0.62
        rbox(ax, lx, ly - 0.22, 0.44, 0.44, col[s_][0], r=0.06, z=3)
        ax.text(lx + 0.65, ly, lab, ha="left", va="center", fontsize=14, color=TEXT_COLOR)

    x = pad + w2 + gap
    card(ax, x, cy, w2, ch, "Contact prediction", "contact_prediction",
         "precision @ L, L/2, L/5  ·  short / medium / long range")
    ms = 1.9
    mx, my = x + 1.0, cy + 0.95
    contact_map(ax, mx, my, ms)
    tx = mx + ms + 0.6
    ax.text(tx, my + ms - 0.1, "low-rank bilinear head on residue pairs", ha="left", va="center", fontsize=15,
            color=TEXT_COLOR)
    rbox(ax, tx, my + ms - 0.95, 0.44, 0.44, MASK_BG, r=0.06, z=3)
    ax.text(tx + 0.65, my + ms - 0.73, "contact in 3D, |i − j| ≥ 6", ha="left", va="center", fontsize=14,
            color=TEXT_COLOR)
    rbox(ax, tx, my + ms - 1.6, 0.44, 0.44, RESIDUE_BG, r=0.06, z=3)
    ax.text(tx + 0.65, my + ms - 1.38, "near-diagonal pairs, excluded", ha="left", va="center", fontsize=14,
            color=TEXT_COLOR)

    # ---------------------------------------------------------------- lane 3: zero-shot
    cy = y3 + pady
    x = pad
    card(ax, x, cy, w2, ch, "Pseudo-perplexity", "pseudo_perplexity",
         "")
    ax.text(x + 0.4, cy + 0.5, r"PPPL $= \exp\left(-\,\mathrm{mean}_i\ \log p(x_i \mid x_{\setminus i})\right)$"
            "   ·  lower is better", ha="left", va="center", fontsize=15, color=GREY)
    seq = "MQIFVKTL"
    rx = x + 0.4
    for r, (pos, aa) in enumerate([(0, "M"), (3, "F")]):
        ry = cy + 1.0 + (1 - r) * (C + 0.12)
        e = cells(ax, rx, ry, seq, fs=14, highlight={pos: (MASK_BG, "<mask>", "white")})
        ax.text(e + 0.3, ry + C / 2, f"→ log p({aa})", ha="left", va="center", fontsize=15, color=TEXT_COLOR)
    ax.text(e + 3.0, cy + 1.0 + (C + 0.12) / 2 + C / 2, "… once per residue", ha="left", va="center",
            fontsize=15, color=GREY, style="italic")

    x = pad + w2 + gap
    card(ax, x, cy, w2, ch, "Categorical Jacobian", "categorical_jacobian",
         "mutate each residue  ·  logit change elsewhere → contacts")
    ry = cy + 1.25
    rx = x + 0.4
    e = cells(ax, rx, ry, seq, highlight={3: (GREEN, "W", TEXT_COLOR)})
    ax.text(rx + 3 * (C + G) + C / 2, ry - 0.3, "F→W", ha="center", va="center", fontsize=13, color=GREY)
    heights = [0.15, 0.25, 0.5, 0, 0.55, 0.2, 0.1, 0.45]
    for k, hgt in enumerate(heights):
        if k == 3:
            continue
        rbox(ax, rx + k * (C + G) + 0.2, ry + C + 0.1, C - 0.4, hgt + 0.05, MASK_BG, r=0.03, z=3)
    arrow(ax, e + 0.3, e + 1.3, ry + C / 2)
    ax.text(e + 0.8, ry + C / 2 + 0.4, "Δ logits", ha="center", va="center", fontsize=13, color=GREY)
    contact_map(ax, e + 1.9, cy + 0.95, 1.9)

    save(fig, out, transparent, dpi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    p.add_argument("--opaque", action="store_true")
    p.add_argument("--dpi", type=int, default=340)
    a = p.parse_args()
    plot(a.outdir / "downstream_tasks.png", not a.opaque, a.dpi)


if __name__ == "__main__":
    main()
