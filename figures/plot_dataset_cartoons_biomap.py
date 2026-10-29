"""Cartoons of the 13 Biomap-research datasets: what each one asks about a protein.

Mirrors modules/evaluate/scripts/dataset_sourcing/configs/source_biomap_datasets.yaml (dataset list),
configs/datasets/<name>.yaml (task per dataset) and the upstream biomap-research dataset cards
(label meanings). Readout chips are illustrative example values; purple chips are classes,
green chips are numbers. Fold icons, secondary-structure letters, GB1 variants and their bar heights,
and the contact map are schematic.

    python figures/plot_dataset_cartoons_biomap.py --outdir figures
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrow, Wedge

from plot_data_pipeline import GREEN, GREY, LANE1_BG, LANE2_BG, harrow, new_axes, rbox, save
from plot_dataset_cartoons_uniprot_residue import (
    GAP, IRIS, PAD, PH, PW, SUB, blob, label, panel)
from plot_downstream_tasks import COIL, MONO, UBIQUITIN, cells, contact_map, lane
from plot_masked_sequence import MASK_BG, RESIDUE_BG, TEXT_COLOR
from plot_pretrain_components import NEUTRAL_BG

CHIP_FS = 15


def readout(ax, x, y, text, fc=MASK_BG, tc="white"):
    w = 0.2 * len(text) * CHIP_FS / 17 + 1.0
    rbox(ax, x + (PW - w) / 2, y + 0.25, w, 0.75, fc, r=0.2, z=3)
    ax.text(x + PW / 2, y + 0.625, text, ha="center", va="center", fontsize=CHIP_FS, fontweight="bold",
            color=tc, zorder=4)


def number(ax, x, y, text):
    readout(ax, x, y, text, GREEN, TEXT_COLOR)


# --------------------------------------------------------------------------- per protein: class

def fold_prediction(ax, x, y):
    panel(ax, x, y, "Fold")
    cy = y + 2.6
    for k in range(3):                                         # helix bundle
        rbox(ax, x + 0.55 + k * 0.45, cy - 0.75, 0.32, 1.5, COIL, r=0.14, z=3)
    for k in range(4):                                         # antiparallel beta sheet
        x0, up = x + 2.6 + k * 0.45, k % 2 == 0
        ax.add_patch(FancyArrow(x0, cy - 0.75 if up else cy + 0.75, 0, 1.5 if up else -1.5, width=0.26,
                                head_width=0.42, head_length=0.3, length_includes_head=True, fc=MASK_BG,
                                ec="none", zorder=3))
    bx = x + 5.35                                              # beta barrel seen end-on
    for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
        ax.add_patch(Circle((bx + 0.6 * np.cos(a), cy + 0.6 * np.sin(a)), 0.17, fc=COIL, ec="none", zorder=3))
    readout(ax, x, y, "fold 1 of 1,195")


def localization_prediction(ax, x, y):
    panel(ax, x, y, "Subcellular localization")
    cx, cy = x + PW / 2, y + 2.65
    ax.add_patch(Ellipse((cx, cy), 5.6, 2.9, fc=NEUTRAL_BG, ec=GREY, lw=2, zorder=2))
    ax.add_patch(Circle((cx - 1.2, cy), 0.85, fc=LANE1_BG, ec=IRIS, lw=2, zorder=3))
    ax.add_patch(Ellipse((cx + 1.35, cy + 0.45), 1.5, 0.75, fc=LANE2_BG, ec=GREEN, lw=2, zorder=3))
    t = np.linspace(0, 1, 60)
    ax.plot(cx + 0.75 + 1.2 * t, cy + 0.45 + 0.17 * np.sin(2 * np.pi * 3 * t), color=GREEN, lw=1.6, zorder=3.5)
    for dx, dy in ((-0.25, 0.3), (0.2, 0.1), (-0.1, -0.35), (0.35, -0.3)):
        ax.add_patch(Circle((cx - 1.2 + dx, cy + dy), 0.15, fc=MASK_BG, ec="none", zorder=4))
    for dx, dy in ((1.0, -0.6), (1.7, -0.45), (0.4, 0.9)):
        ax.add_patch(Circle((cx + dx, cy + dy), 0.15, fc=COIL, ec="none", zorder=4))
    readout(ax, x, y, "nucleus")


def metal_ion_binding(ax, x, y):
    panel(ax, x, y, "Metal ion binding")
    cx, cy, ang = x + 3.0, y + 2.55, np.pi / 4
    blob(ax, cx, cy, 1.25, RESIDUE_BG, seed=0.4, pocket=(ang, 0.4))
    ix, iy = cx + 1.05 * np.cos(ang), cy + 0.95 * np.sin(ang)
    ax.add_patch(Circle((ix, iy), 0.52, fc=IRIS, ec="white", lw=1.5, zorder=4))
    ax.text(ix, iy, "Zn²⁺", ha="center", va="center", fontsize=14, fontweight="bold", color="white", zorder=5)
    readout(ax, x, y, "binds metal: yes")


def solubility_prediction(ax, x, y):
    panel(ax, x, y, "Solubility")
    cy = y + 2.7
    for k, (dx, dy) in enumerate(((0.6, 0.5), (1.6, 0.8), (1.1, -0.2), (2.2, 0.0), (0.5, -0.6), (1.9, -0.8))):
        blob(ax, x + dx, cy + dy, 0.3, MASK_BG, seed=k)
    for k, (dx, dy) in enumerate(((0, 0), (0.42, 0.15), (-0.38, 0.22), (0.1, -0.38), (0.5, -0.3), (-0.3, -0.3),
                                  (0.05, 0.45))):
        blob(ax, x + 4.9 + dx, cy + dy, 0.3, COIL, seed=k + 3, z=2 + k * 0.01)
    label(ax, x + 1.4, y + 1.45, "soluble")
    label(ax, x + 4.9, y + 1.45, "aggregates")
    readout(ax, x, y, "soluble")


def temperature_stability(ax, x, y):
    panel(ax, x, y, "Temperature stability")
    tx, ty = x + 1.4, y + 1.85
    rbox(ax, tx - 0.17, ty, 0.34, 2.0, "white", ec=GREY, r=0.17, lw=2, z=3)
    rbox(ax, tx - 0.1, ty, 0.2, 1.45, MASK_BG, r=0.1, z=3.5)
    ax.add_patch(Circle((tx, ty), 0.36, fc=MASK_BG, ec=GREY, lw=2, zorder=4))
    label(ax, tx + 0.35, ty + 1.45, "65 °C", ha="left", color=TEXT_COLOR, fontweight="bold")
    blob(ax, x + 4.6, y + 2.6, 1.0, RESIDUE_BG, seed=1.7)
    label(ax, x + 4.6, y + 1.35, "still folded?")
    readout(ax, x, y, "stable at 65 °C")


# --------------------------------------------------------------------------- per protein: number

def enzyme_catalytic_efficiency(ax, x, y):
    panel(ax, x, y, "Catalytic rate (kcat)")
    cx, cy = x + 2.0, y + 2.6
    blob(ax, cx, cy, 1.15, RESIDUE_BG, seed=2.2, pocket=(0.0, 0.45))
    ax.add_patch(Circle((cx + 0.85, cy), 0.3, fc=GREEN, ec="white", lw=1.2, zorder=4))
    harrow(ax, x + 3.6, x + 4.5, cy)
    ax.add_patch(Wedge((x + 4.95, cy), 0.3, 90, 270, fc=GREEN, ec="white", lw=1.2, zorder=4))
    ax.add_patch(Wedge((x + 5.35, cy), 0.3, 270, 90, fc=GREEN, ec="white", lw=1.2, zorder=4))
    label(ax, x + 4.05, cy + 0.5, "per s")
    number(ax, x, y, "turnovers per second")


def fitness_prediction(ax, x, y):
    panel(ax, x, y, "Fitness (GB1)")
    cy = y + 2.6
    # IgG Fc fragment: two heavy-chain halves
    rbox(ax, x + 0.4, cy - 0.85, 0.5, 1.7, IRIS, r=0.2, z=3)
    rbox(ax, x + 0.95, cy - 0.85, 0.5, 1.7, IRIS, r=0.2, z=3)
    label(ax, x + 0.92, cy - 1.15, "IgG Fc")
    gx = x + 2.45
    blob(ax, gx, cy, 0.85, RESIDUE_BG, seed=1.4)
    for dx, dy in ((-0.5, 0.35), (-0.55, -0.05), (-0.45, -0.42), (-0.1, 0.5)):
        ax.add_patch(Circle((gx + dx, cy + dy), 0.13, fc=MASK_BG, ec="white", lw=1, zorder=4))
    label(ax, gx, cy - 1.15, "GB1")
    label(ax, gx, cy + 1.15, "4 mutated sites", color=MASK_BG, fontweight="bold")
    bx, base = x + 3.75, cy - 0.85
    for k, (seq, h) in enumerate((("VDGV", 1.0), ("FWAV", 1.6), ("VDPA", 0.3))):
        rbox(ax, bx + k * 0.95, base, 0.55, h, GREEN, r=0.05, z=3)
        ax.text(bx + k * 0.95 + 0.27, base - 0.3, seq, ha="center", va="center", fontsize=14, family=MONO,
                color=SUB, zorder=4)
    label(ax, bx + 1.22, cy + 1.15, "binding")
    number(ax, x, y, "fitness = how well it binds")


def activity_curve(ax, x, y, peak, ticks, unit):
    x0, x1, base = x + 0.8, x + 6.0, y + 1.8
    ax.plot([x0, x1], [base, base], color=GREY, lw=1.8, zorder=3)
    ax.plot([x0, x0], [base, base + 2.1], color=GREY, lw=1.8, zorder=3)
    xs = np.linspace(x0, x1, 200)
    ax.plot(xs, base + 1.8 * np.exp(-((xs - peak) / 0.95) ** 2), color=MASK_BG, lw=3.2, zorder=4)
    ax.plot([peak, peak], [base, base + 1.8], color=IRIS, lw=1.8, ls=(0, (3, 2)), zorder=3.5)
    for tx, text in ticks:
        label(ax, tx, base - 0.3, text)
    ax.text(x0 - 0.3, base + 1.0, "activity", ha="center", va="center", fontsize=14, color=SUB, rotation=90)
    label(ax, x1, base + 0.3, unit, ha="right")


def optimal_ph(ax, x, y):
    panel(ax, x, y, "Optimal pH")
    activity_curve(ax, x, y, x + 3.2, ((x + 1.3, "4"), (x + 3.4, "7"), (x + 5.5, "10")), "pH")
    number(ax, x, y, "pH 6.5")


def optimal_temperature(ax, x, y):
    panel(ax, x, y, "Optimal temperature")
    activity_curve(ax, x, y, x + 4.0, ((x + 1.3, "20"), (x + 3.4, "50"), (x + 5.5, "80")), "°C")
    number(ax, x, y, "62 °C")


def protease(ax, x, y):
    ax.add_patch(Wedge((x, y), 0.38, 210, 150, fc=IRIS, ec="none", zorder=3))


def stability_prediction(ax, x, y):
    panel(ax, x, y, "Stability (protease)")
    r1, r2 = y + 3.2, y + 1.85
    label(ax, x + 2.2, r1 + 0.75, "protease", color=IRIS, fontweight="bold")
    # folded: protease cannot reach the backbone
    blob(ax, x + 0.85, r1, 0.45, RESIDUE_BG, seed=0.9)
    protease(ax, x + 2.2, r1)
    harrow(ax, x + 2.75, x + 3.4, r1, color=GREY)
    blob(ax, x + 3.95, r1, 0.45, RESIDUE_BG, seed=0.9)
    label(ax, x + 4.65, r1, "folded:", ha="left", color=TEXT_COLOR)
    label(ax, x + 4.65, r1 - 0.42, "survives", ha="left")
    # unfolded: chain is cut into pieces
    t = np.linspace(0, 1, 60)
    ax.plot(x + 0.35 + 1.0 * t, r2 + 0.15 * np.sin(2 * np.pi * 3 * t), color=COIL, lw=3, zorder=3)
    protease(ax, x + 2.2, r2)
    harrow(ax, x + 2.75, x + 3.4, r2, color=GREY)
    for k in range(3):
        ax.plot(x + 3.5 + k * 0.33 + 0.2 * t, r2 + 0.12 * np.sin(2 * np.pi * t + k), color=COIL, lw=3,
                zorder=3)
    label(ax, x + 4.65, r2, "unfolded:", ha="left", color=TEXT_COLOR)
    label(ax, x + 4.65, r2 - 0.42, "cut", ha="left")
    number(ax, x, y, "protease resistance")


# --------------------------------------------------------------------------- per residue / pair

SS_COLORS = {"H": (MASK_BG, "white"), "G": (MASK_BG, "white"), "I": (MASK_BG, "white"),
             "E": (GREEN, TEXT_COLOR), "B": (GREEN, TEXT_COLOR),
             "T": (COIL, "white"), "S": (COIL, "white"), "C": (COIL, "white")}


def ss_panel(ax, x, y, title, ss, legend, columns=1):
    """legend = [(letter, meaning)], laid out in columns below the residue tiles."""
    panel(ax, x, y, title)
    seg = UBIQUITIN[16:16 + len(ss)]
    ty = y + 3.25
    cells(ax, x + 0.45, ty, seg)
    cells(ax, x + 0.45, ty - 0.92, ss,
          highlight={k: (SS_COLORS[s][0], s, SS_COLORS[s][1]) for k, s in enumerate(ss)})
    rows = -(-len(legend) // columns)
    step = 0.48 if rows > 3 else 0.55
    for k, (s, meaning) in enumerate(legend):
        lx = x + 0.45 + (k // rows) * 3.05
        ly = y + 1.85 - (k % rows) * step
        fc, tc = SS_COLORS[s]
        rbox(ax, lx, ly - 0.21, 0.42, 0.42, fc, r=0.06, z=3)
        ax.text(lx + 0.21, ly, s, ha="center", va="center", fontsize=14, fontweight="bold", color=tc,
                family=MONO, zorder=4)
        label(ax, lx + 0.6, ly, meaning, ha="left", color=TEXT_COLOR)


def ssp_q3(ax, x, y):
    ss_panel(ax, x, y, "Secondary structure Q3", "ECCCCCH",
             [("H", "helix (α, 3₁₀, π)"), ("E", "strand (β-strand, β-bridge)"), ("C", "coil (turn, bend, loop)")])


def ssp_q8(ax, x, y):
    ss_panel(ax, x, y, "Secondary structure Q8", "BSTTSGH",
             [("H", "α-helix"), ("G", "3₁₀-helix"), ("I", "π-helix"), ("T", "turn"),
              ("E", "β-strand"), ("B", "β-bridge"), ("S", "bend"), ("C", "coil")], columns=2)


def contact_prediction_binary(ax, x, y):
    panel(ax, x, y, "Residue contacts")
    contact_map(ax, x + 0.85, y + 1.3, 2.6)
    lx = x + 3.9
    rbox(ax, lx, y + 3.0, 0.4, 0.4, MASK_BG, r=0.06, z=3)
    label(ax, lx + 0.55, y + 3.2, "< 8 Å in 3D", ha="left", color=TEXT_COLOR)
    rbox(ax, lx, y + 2.25, 0.4, 0.4, RESIDUE_BG, r=0.06, z=3)
    label(ax, lx + 0.55, y + 2.45, "|i − j| < 6", ha="left", color=TEXT_COLOR)
    label(ax, lx + 0.55, y + 2.05, "ignored", ha="left")
    readout(ax, x, y, "contact pairs")


ROWS = [
    ("Per protein  ·  class", LANE1_BG, "sequence_classification",
     [fold_prediction, localization_prediction, metal_ion_binding, solubility_prediction, temperature_stability]),
    ("Per protein  ·  number", LANE2_BG, "sequence_regression",
     [enzyme_catalytic_efficiency, fitness_prediction, optimal_ph, optimal_temperature, stability_prediction]),
    ("Per residue  ·  per residue pair", LANE1_BG, "token_classification  ·  contact_prediction",
     [ssp_q3, ssp_q8, contact_prediction_binary]),
]


def plot(out: Path, transparent: bool, dpi: int):
    ncol = 5
    W = 2 * PAD + ncol * PW + (ncol - 1) * GAP
    lane_h = PH + 2 * 0.3
    step = lane_h + 0.85 + 0.35
    fig, ax = new_axes(-0.2, W + 0.2, -0.2, len(ROWS) * step - 0.15)
    for r, (title, bg, task, panels) in enumerate(ROWS):
        ly = (len(ROWS) - 1 - r) * step
        lane(ax, 0, ly, W, lane_h, bg, title, task)
        for c, draw in enumerate(panels):
            draw(ax, PAD + c * (PW + GAP), ly + 0.3)
    save(fig, out, transparent, dpi)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    p.add_argument("--opaque", action="store_true")
    p.add_argument("--dpi", type=int, default=270)
    a = p.parse_args()
    plot(a.outdir / "dataset_cartoons_biomap.png", not a.opaque, a.dpi)


if __name__ == "__main__":
    main()
