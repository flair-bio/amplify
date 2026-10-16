"""Cartoons of the 12 UniProt residue-annotation datasets: what each per-residue label marks.

Mirrors modules/evaluate/scripts/dataset_sourcing/source_uniprot_annotations.py (feature
groups) and configs/datasets/uniprot_*.yaml. Purple beads are residues that get a non-background
class in that dataset; grey beads are background. Shapes are schematic, not to scale.
Also defines the drawing primitives (beads, blobs, helices, membrane, moieties) shared by the
other plot_dataset_cartoons_*.py scripts.

    python figures/plot_dataset_cartoons_uniprot_residue.py --outdir figures
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, FancyArrow, Polygon, Rectangle

from plot_data_pipeline import GREEN, LANE1_BG, LANE2_BG, new_axes, rbox, save
from plot_downstream_tasks import COIL, lane
from plot_masked_sequence import MASK_BG, RESIDUE_BG, TEXT_COLOR

SUB = "#444444"
IRIS = "#614972"
FS = 15
TITLE_FS = 18
BEAD_R = 0.24
PW, PH = 6.6, 5.0          # panel size
GAP = 0.5
PAD = 0.5


# --------------------------------------------------------------------------- shared primitives

def panel(ax, x, y, title):
    rbox(ax, x, y, PW, PH, "white", r=0.25, z=1)
    ax.text(x + 0.3, y + PH - 0.45, title, ha="left", va="center", fontsize=TITLE_FS, fontweight="bold",
            color=TEXT_COLOR, zorder=5)


def label(ax, x, y, text, ha="center", color=SUB, **kw):
    ax.text(x, y, text, ha=ha, va="center", fontsize=FS, color=color, zorder=6, **kw)


def backbone(ax, pts, color=COIL, lw=2.4, z=2):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="round", solid_joinstyle="round")


def beads(ax, pts, marks=None, r=BEAD_R, line=True):
    """Backbone through pts with one bead per point; marks = {index: (facecolor, letter)}."""
    marks = marks or {}
    if line:
        backbone(ax, pts)
    for k, (px, py) in enumerate(pts):
        fc, letter = marks.get(k, (RESIDUE_BG, ""))
        rr = r * 1.25 if letter else r
        ax.add_patch(Circle((px, py), rr, fc=fc, ec="white", lw=1.2, zorder=3))
        if letter:
            ax.text(px, py, letter, ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if fc != RESIDUE_BG else TEXT_COLOR, zorder=4)


def row(x0, x1, y, n, amp=0.0, waves=1.0):
    xs = np.linspace(x0, x1, n)
    return [(x, y + amp * np.sin(2 * np.pi * waves * (x - x0) / (x1 - x0))) for x in xs]


def blob(ax, cx, cy, r, fc=RESIDUE_BG, seed=0.0, pocket=None, z=2, squash=0.9):
    """Irregular globule; pocket = (angle, depth) carves a notch (binding cleft)."""
    th = np.linspace(0, 2 * np.pi, 300)
    rr = r * (1 + 0.07 * np.sin(3 * th + seed) + 0.05 * np.cos(5 * th + 2 * seed))
    if pocket:
        ang, depth = pocket
        d = np.angle(np.exp(1j * (th - ang)))
        rr = rr * (1 - depth * np.exp(-(d / 0.35) ** 2))
    ax.fill(cx + rr * np.cos(th), cy + squash * rr * np.sin(th), fc=fc, ec="none", zorder=z)


def helix(ax, p0, p1, turns, color=MASK_BG, r=0.3, lw=3.2, z=3):
    """Cartoon spring from p0 to p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    length = np.hypot(*d)
    u = d / length
    v = np.array([-u[1], u[0]])
    t = np.linspace(0, 2 * np.pi * turns, 400)
    a = length / (2 * np.pi * turns)
    along = a * t - 1.6 * a * np.sin(t)
    perp = -r * np.cos(t)
    pts = p0 + np.outer(along, u) + np.outer(perp, v)
    pts = np.vstack([p0, pts, p1])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=z, solid_capstyle="round")


def strand(ax, x0, x1, y, color=GREEN, z=3):
    ax.add_patch(FancyArrow(x0, y, x1 - x0, 0, width=0.34, head_width=0.66, head_length=0.42,
                            length_includes_head=True, fc=color, ec="none", zorder=z))


def membrane(ax, x0, x1, yb, h=1.1):
    ax.add_patch(Rectangle((x0, yb), x1 - x0, h, fc=LANE2_BG, ec="none", zorder=1.5))
    for x in np.arange(x0 + 0.13, x1, 0.27):
        for yy in (yb, yb + h):
            ax.add_patch(Circle((x, yy), 0.12, fc=GREEN, ec="none", zorder=1.6))


def phosphate(ax, x, y, bx, by, z=4):
    ax.plot([bx, x], [by, y], color=IRIS, lw=2.2, zorder=z - 1)
    ax.add_patch(Circle((x, y), 0.3, fc=IRIS, ec="white", lw=1.2, zorder=z))
    ax.text(x, y, "P", ha="center", va="center", fontsize=14, fontweight="bold", color="white", zorder=z + 1)


def glycan(ax, bx, by, scale=1.0, branched=True):
    """Schematic glycan tree: squares (GlcNAc-like) then a branching circle layer (mannose-like)."""
    s = 0.34 * scale
    step = 0.48 * scale
    nodes = [(bx, by + step + 0.1, "sq"), (bx, by + 2 * step + 0.1, "sq")]
    edges = [((bx, by), nodes[0][:2]), (nodes[0][:2], nodes[1][:2])]
    if branched:
        top = (bx, by + 3 * step + 0.1)
        nodes += [(*top, "ci"), (bx - 0.42 * scale, top[1] + step * 0.9, "ci"),
                  (bx + 0.42 * scale, top[1] + step * 0.9, "ci")]
        edges += [(nodes[1][:2], top), (top, nodes[3][:2]), (top, nodes[4][:2])]
    for (xa, ya), (xb, yb) in edges:
        ax.plot([xa, xb], [ya, yb], color=IRIS, lw=2, zorder=3.5)
    for nx, ny, kind in nodes:
        if kind == "sq":
            ax.add_patch(Rectangle((nx - s / 2, ny - s / 2), s, s, fc=IRIS, ec="white", lw=1, zorder=4))
        else:
            ax.add_patch(Circle((nx, ny), s / 2, fc=GREEN, ec="white", lw=1, zorder=4))


def lipid(ax, bx, by, top, z=3.5):
    """Zig-zag acyl chain from a bead up to y=top."""
    ys = np.arange(by + 0.25, top, 0.17)
    xs = bx + 0.1 * np.where(np.arange(len(ys)) % 2, 1, -1)
    ax.plot([bx, *xs], [by, *ys], color=IRIS, lw=2.6, zorder=z, solid_joinstyle="miter")


def bracket(ax, x0, x1, y, text, down=False):
    tick = -0.18 if down else 0.18
    ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick], color=SUB, lw=1.6, zorder=4)
    label(ax, (x0 + x1) / 2, y + (-0.35 if down else 0.35), text)


def legend_dot(ax, x, y, fc, text):
    ax.add_patch(Circle((x, y), 0.17, fc=fc, ec="none", zorder=4))
    label(ax, x + 0.3, y, text, ha="left")


# --------------------------------------------------------------------------- panels

def secondary_structure(ax, x, y):
    panel(ax, x, y, "Secondary structure")
    yb = y + 2.2
    backbone(ax, [(x + 0.4, yb), (x + 6.2, yb)])
    beads(ax, row(x + 0.45, x + 0.95, yb, 2), line=False)
    helix(ax, (x + 1.2, yb), (x + 2.9, yb), 4)
    beads(ax, row(x + 3.1, x + 3.4, yb, 2), line=False)
    strand(ax, x + 3.55, x + 5.0, yb)
    th = np.linspace(np.pi, 0, 40)
    ax.plot(x + 5.5 + 0.35 * np.cos(th), yb + 0.7 * np.sin(th), color=IRIS, lw=3.2, zorder=3)
    label(ax, x + 2.05, yb - 0.9, "helix")
    label(ax, x + 4.25, yb - 0.9, "strand")
    label(ax, x + 5.5, yb - 0.9, "turn")


def domains(ax, x, y):
    panel(ax, x, y, "Domains")
    yb = y + 2.1
    backbone(ax, [(x + 0.4, yb), (x + 6.2, yb)])
    beads(ax, row(x + 0.45, x + 0.95, yb, 2), line=False)
    blob(ax, x + 2.05, yb + 0.1, 0.95, MASK_BG, seed=0.5, z=2.5)
    beads(ax, row(x + 3.15, x + 3.65, yb, 2), line=False)
    blob(ax, x + 4.65, yb + 0.1, 0.85, MASK_BG, seed=2.0, z=2.5)
    beads(ax, row(x + 5.7, x + 6.15, yb, 2), line=False)
    label(ax, x + 2.05, y + 0.6, "domain")
    label(ax, x + 4.65, y + 0.6, "domain")


def regions(ax, x, y):
    panel(ax, x, y, "Regions")
    pts = row(x + 0.45, x + 6.15, y + 1.7, 14, amp=0.35, waves=2.0)
    beads(ax, pts, {k: (MASK_BG, "") for k in range(4, 10)})
    bracket(ax, pts[4][0] - 0.2, pts[9][0] + 0.2, y + 2.65, "e.g. binds a partner")


def structures(ax, x, y):
    panel(ax, x, y, "Repeats & motifs")
    cy = y + 2.1
    # repeat: identical units in tandem
    backbone(ax, [(x + 0.35, cy), (x + 2.25, cy)])
    for k in range(3):
        rbox(ax, x + 0.45 + k * 0.62, cy - 0.45, 0.45, 0.9, MASK_BG, r=0.12, z=3)
    label(ax, x + 1.3, y + 0.6, "repeat")
    # coiled coil: two helices wound round each other
    t = np.linspace(0, 1, 200)
    for ph, col in ((0, MASK_BG), (np.pi, IRIS)):
        ax.plot(x + 2.65 + 1.5 * t, cy + 0.32 * np.sin(2 * np.pi * 2.5 * t + ph), color=col, lw=4, zorder=3)
    label(ax, x + 3.4, y + 0.6, "coiled coil")
    # zinc finger: loop of residues gripping a zinc ion
    zx, zy, rr = x + 5.45, cy, 0.72
    ang = np.linspace(-np.pi / 2, 3 * np.pi / 2, 10)[:-1]
    loop = [(zx + rr * np.cos(a), zy + rr * np.sin(a)) for a in ang]
    ligands = (1, 3, 5, 7)
    for k in ligands:
        ax.plot([zx, loop[k][0]], [zy, loop[k][1]], color=IRIS, lw=1.6, ls=(0, (2, 1.5)), zorder=2.5)
    beads(ax, loop, {k: (MASK_BG, "") for k in range(len(loop))}, r=0.17)
    ax.add_patch(Circle((zx, zy), 0.3, fc=IRIS, ec="white", lw=1.2, zorder=4))
    ax.text(zx, zy, "Zn", ha="center", va="center", fontsize=14, fontweight="bold", color="white", zorder=5)
    label(ax, zx, y + 0.6, "zinc finger")


def membrane_pass(ax, x, y):
    panel(ax, x, y, "Membrane passes")
    mb, mh = y + 1.35, 1.3
    membrane(ax, x + 0.15, x + PW - 0.15, mb, mh)
    top = mb + mh + 0.6
    beads(ax, row(x + 0.5, x + 1.2, top, 2))
    backbone(ax, [(x + 1.2, top), (x + 1.6, mb + mh + 0.2)])
    helix(ax, (x + 1.6, mb + mh + 0.2), (x + 1.6, mb - 0.2), 4)
    below = [(x + 1.6, mb - 0.2), (x + 2.2, mb - 0.6), (x + 2.9, mb - 0.6), (x + 3.5, mb - 0.2)]
    beads(ax, below[1:-1])
    backbone(ax, below)
    # re-entrant (intramembrane) segment: dips into the bilayer and returns to the same side
    helix(ax, (x + 3.5, mb - 0.2), (x + 4.05, mb + 0.9), 2, color=IRIS, r=0.2)
    backbone(ax, [(x + 4.05, mb + 0.9), (x + 4.6, mb - 0.2), (x + 5.1, mb - 0.6)])
    beads(ax, row(x + 5.1, x + 6.1, mb - 0.6, 3), line=False)
    label(ax, x + 2.1, top, "transmembrane", ha="left")
    label(ax, x + 4.1, y + 0.3, "intramembrane")


def topology(ax, x, y):
    panel(ax, x, y, "Topology")
    mb, mh = y + 1.45, 1.1
    membrane(ax, x + 0.15, x + PW - 0.15, mb, mh)
    top, bot = mb + mh + 0.7, mb - 0.55
    out_loop = row(x + 2.1, x + 3.6, top, 4, amp=0.2)
    in_l = row(x + 0.4, x + 1.3, bot, 3)
    in_r = row(x + 4.4, x + 5.3, bot, 3)
    helix(ax, (x + 1.3, bot), (x + 2.1, top), 3, color=COIL)
    helix(ax, (x + 3.6, top), (x + 4.4, bot), 3, color=COIL)
    beads(ax, out_loop, {k: (GREEN, "") for k in range(4)})
    beads(ax, in_l, {k: (MASK_BG, "") for k in range(3)})
    beads(ax, in_r, {k: (MASK_BG, "") for k in range(3)})
    label(ax, x + PW - 0.3, top + 0.1, "extracellular", ha="right")
    label(ax, x + 3.3, y + 0.45, "cytoplasmic")


def peptide(ax, x, y):
    panel(ax, x, y, "Signal & transit peptides")
    yb = y + 2.1
    pts = row(x + 0.45, x + 2.6, yb, 6)
    beads(ax, pts, {k: (MASK_BG, "") for k in range(5)})
    cut = x + 3.0
    ax.plot([cut, cut], [yb - 0.7, yb + 0.7], color=IRIS, lw=2.2, ls=(0, (3, 2)), zorder=4)
    backbone(ax, [pts[-1], (x + 4.3, yb)])
    blob(ax, x + 4.9, yb + 0.1, 0.9, RESIDUE_BG, seed=1.3)
    bracket(ax, pts[0][0] - 0.2, pts[4][0] + 0.2, yb - 0.6, "signal peptide", down=True)
    label(ax, cut, yb + 1.05, "cleaved")
    label(ax, x + 4.9, y + 0.6, "mature protein")


def functional_sites(ax, x, y):
    panel(ax, x, y, "Functional sites")
    cx, cy = x + 2.7, y + 2.0
    blob(ax, cx, cy, 1.5, RESIDUE_BG, seed=0.8, pocket=(np.pi / 2.6, 0.45))
    lig = [(cx + 0.55 + 0.32 * np.cos(a), cy + 1.0 + 0.32 * np.sin(a)) for a in np.arange(6) * np.pi / 3]
    ax.add_patch(Polygon(lig, closed=True, fc=GREEN, ec="white", lw=1.2, zorder=4))
    rim = [(cx + 0.05, cy + 0.95), (cx + 1.1, cy + 0.6)]
    for px, py in rim:
        ax.add_patch(Circle((px, py), BEAD_R, fc=MASK_BG, ec="white", lw=1.2, zorder=4))
    ax.add_patch(Circle((cx + 0.45, cy + 0.45), BEAD_R, fc=IRIS, ec="white", lw=1.2, zorder=4))
    legend_dot(ax, x + 4.55, y + 2.9, MASK_BG, "binding")
    legend_dot(ax, x + 4.55, y + 2.3, IRIS, "active")
    legend_dot(ax, x + 4.55, y + 1.7, GREEN, "ligand")


def glycosylation(ax, x, y):
    panel(ax, x, y, "Glycosylation")
    yb = y + 1.2
    pts = row(x + 0.45, x + 6.15, yb, 10)
    beads(ax, pts, {2: (MASK_BG, "N"), 7: (MASK_BG, "T")})
    glycan(ax, *pts[2])
    glycan(ax, *pts[7], branched=False)
    label(ax, pts[2][0] + 0.75, yb + 2.0, "N-linked", ha="left")
    label(ax, pts[7][0] - 0.4, yb + 1.0, "O-linked", ha="right")


def phosphorylation(ax, x, y):
    panel(ax, x, y, "Phosphorylation")
    yb = y + 1.4
    pts = row(x + 0.45, x + 6.15, yb, 10, amp=0.2)
    beads(ax, pts, {2: (MASK_BG, "S"), 4: (RESIDUE_BG, "T"), 7: (MASK_BG, "Y")})
    for k in (2, 7):
        phosphate(ax, pts[k][0], pts[k][1] + 1.2, *pts[k])
    label(ax, pts[2][0], yb - 0.8, "phospho-Ser")
    label(ax, pts[7][0], yb - 0.8, "phospho-Tyr")


def lipidation(ax, x, y):
    panel(ax, x, y, "Lipidation")
    mb = y + 2.75
    membrane(ax, x + 0.15, x + PW - 0.15, mb, 1.05)
    yb = y + 1.3
    pts = row(x + 0.45, x + 6.15, yb, 10, amp=0.15)
    beads(ax, pts, {3: (MASK_BG, "C")})
    lipid(ax, *pts[3], top=mb + 0.95)
    label(ax, pts[3][0], yb - 0.75, "S-palmitoyl Cys")
    label(ax, x + 5.0, mb + 0.52, "membrane", color=TEXT_COLOR)


def ptm(ax, x, y):
    panel(ax, x, y, "All modifications")
    yb = y + 1.25
    pts = row(x + 0.45, x + 6.15, yb, 12)
    marks = {1: (MASK_BG, "S"), 4: (MASK_BG, "N"), 7: (MASK_BG, "C"), 9: (MASK_BG, "C"), 11: (MASK_BG, "C")}
    beads(ax, pts, marks)
    phosphate(ax, pts[1][0], yb + 1.15, *pts[1])
    glycan(ax, *pts[4], scale=0.8, branched=False)
    lipid(ax, *pts[7], top=yb + 1.9)
    th = np.linspace(0, np.pi, 40)
    mx, half = (pts[9][0] + pts[11][0]) / 2, (pts[11][0] - pts[9][0]) / 2
    ax.plot(mx + half * np.cos(th), yb + 0.3 + 0.9 * np.sin(th), color=IRIS, lw=2.4, zorder=2.5)
    label(ax, mx, yb + 1.5, "S–S")
    for k, text in ((1, "modified"), (4, "glycan"), (7, "lipid")):
        label(ax, pts[k][0], yb - 0.75, text)
    label(ax, mx, yb - 0.75, "disulfide")


ROWS = [
    ("Structure", LANE1_BG, [secondary_structure, domains, regions, structures]),
    ("Membranes & sites", LANE2_BG, [membrane_pass, topology, peptide, functional_sites]),
    ("Chemical modifications", LANE1_BG, [ptm, glycosylation, phosphorylation, lipidation]),
]


def plot(out: Path, transparent: bool, dpi: int):
    ncol = 4
    W = 2 * PAD + ncol * PW + (ncol - 1) * GAP
    lane_h = PH + 2 * 0.3
    step = lane_h + 0.85 + 0.35
    fig, ax = new_axes(-0.2, W + 0.2, -0.2, len(ROWS) * step - 0.15)
    for r, (title, bg, panels) in enumerate(ROWS):
        ly = (len(ROWS) - 1 - r) * step
        lane(ax, 0, ly, W, lane_h, bg, title, "" if r else "coloured = labelled residue  ·  grey = background")
        for c, draw in enumerate(panels):
            draw(ax, PAD + c * (PW + GAP), ly + 0.3)
    save(fig, out, transparent, dpi)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    p.add_argument("--opaque", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    a = p.parse_args()
    plot(a.outdir / "dataset_cartoons_uniprot_residue.png", not a.opaque, a.dpi)


if __name__ == "__main__":
    main()
