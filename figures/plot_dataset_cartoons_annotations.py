"""Cartoons of the protein-level annotation datasets: UniProt GO (3 aspects) and InterPro (7 entry types).

Mirrors modules/evaluate/scripts/dataset_sourcing/source_uniprot_go.py (GO_mf / GO_bp / GO_cc) and
source_interpro.py (entry types) with configs/datasets/{GO_*,interpro_*}.yaml. Both are
sequence_multilabel_classification: one bit per vocabulary entry. Example terms and the bit vector
are illustrative. Produces two figures:

  dataset_cartoons_go.png        one panel per GO aspect
  dataset_cartoons_interpro.png  one protein carrying every InterPro entry type

    python figures/plot_dataset_cartoons_annotations.py --outdir figures
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon

from plot_data_pipeline import GREEN, GREY, LANE1_BG, LANE2_BG, harrow, new_axes, rbox, save
from plot_dataset_cartoons_biomap import readout
from plot_dataset_cartoons_uniprot_residue import (
    GAP, IRIS, PAD, PH, PW, TITLE_FS, blob, bracket, label, panel, phosphate)
from plot_downstream_tasks import MONO, lane
from plot_masked_sequence import MASK_BG, RESIDUE_BG, TEXT_COLOR
from plot_pretrain_components import NEUTRAL_BG

TASK = "sequence_multilabel_classification"


# --------------------------------------------------------------------------- GO

def go_mf(ax, x, y):
    panel(ax, x, y, "Molecular function")
    cx, cy = x + 2.3, y + 2.6
    blob(ax, cx, cy, 1.2, MASK_BG, seed=1.1, pocket=(0.0, 0.4))
    blob(ax, x + 4.9, cy - 0.2, 0.55, RESIDUE_BG, seed=2.5)
    phosphate(ax, x + 4.9, cy + 0.85, x + 4.9, cy + 0.1)
    harrow(ax, cx + 1.0, x + 4.15, cy, color=GREY)
    label(ax, x + 4.9, y + 1.55, "substrate")
    readout(ax, x, y, "kinase · ATP binding · …")


def go_bp(ax, x, y):
    panel(ax, x, y, "Biological process")
    cy = y + 2.6
    xs = [x + 0.8 + k * 1.65 for k in range(4)]
    for k, nx in enumerate(xs):
        ax.add_patch(Circle((nx, cy), 0.35, fc=NEUTRAL_BG, ec=GREY, lw=2, zorder=3))
        if k < 3:
            harrow(ax, nx + 0.45, xs[k + 1] - 0.45, cy, color=GREY)
    blob(ax, (xs[1] + xs[2]) / 2, cy + 0.75, 0.42, MASK_BG, seed=0.3, z=4)
    label(ax, (xs[0] + xs[-1]) / 2, y + 1.55, "one step of a pathway")
    readout(ax, x, y, "signaling · cell cycle · …")


def go_cc(ax, x, y):
    panel(ax, x, y, "Cellular component")
    cx, cy = x + PW / 2, y + 2.6
    ax.add_patch(Ellipse((cx, cy), 5.6, 2.7, fc=NEUTRAL_BG, ec=GREY, lw=3, zorder=2))
    ax.add_patch(Circle((cx - 0.9, cy - 0.1), 0.75, fc=LANE1_BG, ec=IRIS, lw=2, zorder=3))
    for a in (0.35, 0.75, 1.15):
        ax.add_patch(Ellipse((cx + 2.8 * np.cos(a), cy + 1.35 * np.sin(a)), 0.35, 0.55,
                             angle=np.degrees(a) - 90 + 20, fc=MASK_BG, ec="white", lw=1, zorder=4))
    readout(ax, x, y, "plasma membrane · …")


def plot_go(out: Path, transparent: bool, dpi: int):
    W = 2 * PAD + 3 * PW + 2 * GAP
    lane_h = PH + 0.6
    fig, ax = new_axes(-0.2, W + 0.2, -0.2, lane_h + 1.0)
    lane(ax, 0, 0, W, lane_h, LANE1_BG, "Per protein  ·  set of GO terms", TASK)
    for c, draw in enumerate((go_mf, go_bp, go_cc)):
        draw(ax, PAD + c * (PW + GAP), 0.3)
    save(fig, out, transparent, dpi)


# --------------------------------------------------------------------------- InterPro

def site(ax, x, y, fc, text):
    ax.add_patch(Polygon([(x - 0.22, y + 0.38), (x + 0.22, y + 0.38), (x, y)], closed=True, fc=fc, ec="none",
                         zorder=4))
    label(ax, x, y + 0.7, text)


def plot_interpro(out: Path, transparent: bool, dpi: int):
    W = 2 * PAD + 3 * PW + 2 * GAP
    ph = 5.8
    fig, ax = new_axes(-0.2, W + 0.2, -0.2, ph + 1.65)
    lane(ax, 0, 0, W, ph + 0.6, LANE2_BG, "Per protein  ·  set of InterPro entries", TASK)
    x, y, w = PAD, 0.3, W - 2 * PAD
    rbox(ax, x, y, w, ph, "white", r=0.25, z=1)
    ax.text(x + 0.3, y + ph - 0.45, "One protein, every entry type", ha="left", va="center", fontsize=TITLE_FS,
            fontweight="bold", color=TEXT_COLOR, zorder=5)

    x0, x1, by, bh = x + 0.7, x + 14.3, y + 2.0, 0.55
    rbox(ax, x0, by, x1 - x0, bh, RESIDUE_BG, r=0.2, z=2)
    d1, d2 = (x0 + 1.0, x0 + 4.4), (x0 + 6.0, x0 + 9.0)
    for a, b in (d1, d2):
        rbox(ax, a, by - 0.25, b - a, bh + 0.5, MASK_BG, r=0.2, z=3)
        label(ax, (a + b) / 2, by - 0.6, "domain")
    for k in range(3):
        rbox(ax, x0 + 10.0 + k * 1.15, by - 0.15, 0.9, bh + 0.3, IRIS, r=0.12, z=3)
    label(ax, x0 + 11.6, by - 0.6, "repeat")
    site(ax, d1[0] + 0.5, by + bh + 0.3, IRIS, "active site")
    site(ax, d1[0] + 2.9, by + bh + 0.3, GREEN, "binding site")
    site(ax, d2[0] + 1.5, by + bh + 0.3, GREY, "conserved site")
    bracket(ax, x0, x1, by + bh + 1.65, "family")
    bracket(ax, d1[0], d1[1], by - 1.15, "homologous superfamily", down=True)

    harrow(ax, x1 + 0.4, x1 + 1.3, by + bh / 2)
    vx = x1 + 1.55
    rbox(ax, vx, by - 0.1, 4.6, bh + 0.2, LANE1_BG, r=0.15, z=2)
    ax.text(vx + 2.3, by + bh / 2, "[1 0 1 1 0 … 1]", ha="center", va="center", fontsize=16, family=MONO,
            fontweight="bold", color=TEXT_COLOR, zorder=4)
    label(ax, vx + 2.3, by - 0.6, "one bit per entry")
    label(ax, vx + 2.3, by - 1.0, "has it · not where")
    save(fig, out, transparent, dpi)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    p.add_argument("--opaque", action="store_true")
    p.add_argument("--dpi", type=int, default=400)
    a = p.parse_args()
    plot_go(a.outdir / "dataset_cartoons_go.png", not a.opaque, a.dpi)
    plot_interpro(a.outdir / "dataset_cartoons_interpro.png", not a.opaque, a.dpi)


if __name__ == "__main__":
    main()
