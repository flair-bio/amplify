"""Evaluate module: control conditions and the statistics comparing the trained model against them.

Mirrors modules/evaluate/src/utils/controls.py (control conditions), utils/bootstrap.py and
steps/score.py (bootstrap CIs, paired permutation test, Benjamini-Hochberg), and configs/tasks/*.yaml
(per-task default controls). The bar chart on the right is illustrative: majority_class often sits well
above chance on imbalanced tasks, while weight ablations and scrambled labels land near chance.

    python figures/plot_eval_controls.py --outdir figures
"""

import argparse
from pathlib import Path

from plot_data_pipeline import GREEN, LANE1_BG, LANE2_BG, harrow, new_axes, rbox, save
from plot_downstream_tasks import MONO, cells
from plot_masked_sequence import MASK_BG, TEXT_COLOR

SUB = "#444444"
NEUTRAL_BG = "#EEF2EF"
FS = 17

# (name, trunk, head); True = trained weights, False = random.
WEIGHTS = [
    ("random_trunk", False, True),
    ("random_head", True, False),
    ("random_both", False, False),
    ("untrained", None, False),
]

# Residues shuffle after the first; labels permute across examples (same multiset).
SEQUENCE, SCRAMBLED_SEQUENCE = "MKVFTQ", "MQTKFV"
LABELS, SCRAMBLED_LABELS = "ABACBA", "CABAAB"
TILE, TILE_STEP = 0.36, 0.4
# One colour per symbol so the permutation reads at a glance; the fixed first residue stays grey.
TILE_COLORS = {
    "M": "#D9D9D9", "K": "#9EC5E8", "V": "#F6C28B", "F": "#B9A3E3", "T": "#F4A6A6", "Q": "#A8D5A2",
    "A": "#9EC5E8", "B": "#F6C28B", "C": "#F4A6A6",
}

# (label, score, ci_low, ci_high, stars); illustrative only. Weight ablations are not retrained, so
# their arbitrary outputs land near chance, like scrambled_labels; majority_class exploits class imbalance;
# how much scrambled_sequences keeps (composition) is task-dependent.
CHANCE = 0.25
BARS = [
    ("trained", 0.82, 0.79, 0.85, ""),
    ("scrambled_sequences", 0.63, 0.59, 0.67, "***"),
    ("majority_class", 0.45, 0.45, 0.45, "***"),
    ("scrambled_labels", 0.26, 0.23, 0.29, "***"),
    ("random_trunk", 0.24, 0.20, 0.28, "***"),
    ("random_head", 0.22, 0.18, 0.26, "***"),
]
X_MIN, X_MAX = 0.0, 1.0


def panel(ax, x, y, w, h, bg, title, script):
    rbox(ax, x, y, w, h, bg, r=0.3, z=0)
    ax.text(x + 0.5, y + h - 0.6, title, ha="left", va="center", fontsize=23, fontweight="bold", color=TEXT_COLOR)
    if script:
        ax.text(x + w - 0.5, y + h - 0.6, script, ha="right", va="center", fontsize=15, color=SUB, family=MONO)


def block(ax, x, y, w, h, state, label):
    """Trunk/head block: trained (purple), random (light, dashed), or the original checkpoint (green)."""
    if state is None:
        fc, tc, text = GREEN, TEXT_COLOR, "checkpoint"
    elif state:
        fc, tc, text = MASK_BG, "white", label
    else:
        fc, tc, text = "white", SUB, "random"
    rbox(ax, x, y, w, h, fc, r=0.12, z=2)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=14, color=tc,
            fontweight="bold" if state is not False else "normal", style="italic" if state is False else "normal",
            zorder=3)


def tiles(ax, x, y, chars, ec="none"):
    for j, ch in enumerate(chars):
        rbox(ax, x + j * TILE_STEP, y - TILE / 2, TILE, TILE, TILE_COLORS[ch], ec=ec, lw=1.2, r=0.06, z=3)
        ax.text(x + j * TILE_STEP + TILE / 2, y, ch, ha="center", va="center", fontsize=11, family=MONO,
                fontweight="bold", color=TEXT_COLOR, zorder=4)


def before_after(ax, x, y, before, after, ec="none"):
    span = len(before) * TILE_STEP
    tiles(ax, x, y, before, ec)
    harrow(ax, x + span + 0.05, x + span + 0.5, y, color=SUB)
    tiles(ax, x + span + 0.65, y, after, ec)


def group(ax, x, y, w, h, bg, label):
    rbox(ax, x, y, w, h, bg, r=0.25, z=1)
    ax.text(x + 0.45, y + h / 2, label, ha="center", va="center", rotation=90, fontsize=15, color=SUB,
            style="italic", zorder=3)


def plot(out: Path, transparent: bool, dpi: int):
    lw, rw, gap = 11.8, 10.9, 0.5
    W = lw + gap + rw
    H = 11.2
    fig, ax = new_axes(-0.2, W + 0.2, -0.2, H + 0.2)

    # ---------------------------------------------------------------- controls
    panel(ax, 0, 0, lw, H, NEUTRAL_BG, "Controls", "predict.controls")
    gx, gw = 0.4, lw - 0.8
    nx, bx = gx + 1.05, gx + 5.4
    rh = 0.92

    # randomised weights
    gh = len(WEIGHTS) * rh + 0.95
    gy = H - 1.45 - gh
    group(ax, gx, gy, gw, gh, LANE1_BG, "weights")
    ax.text(bx + 1.1, gy + gh - 0.38, "trunk", ha="center", va="center", fontsize=14, color=SUB)
    ax.text(bx + 3.15, gy + gh - 0.38, "head", ha="center", va="center", fontsize=14, color=SUB)
    for k, (name, trunk, head) in enumerate(WEIGHTS):
        yy = gy + gh - 1.15 - k * rh
        ax.text(nx, yy, name, ha="left", va="center", fontsize=16, family=MONO, color=TEXT_COLOR, zorder=3)
        block(ax, bx, yy - 0.33, 2.2, 0.66, trunk, "trained")
        block(ax, bx + 2.35, yy - 0.33, 1.6, 0.66, head, "trained")

    # corrupted inputs / labels
    gh2 = 2 * rh + 0.45
    gy2 = gy - 0.25 - gh2
    group(ax, gx, gy2, gw, gh2, LANE2_BG, "data")
    yy = gy2 + gh2 - 0.68
    ax.text(nx, yy, "scrambled_sequences", ha="left", va="center", fontsize=16, family=MONO, color=TEXT_COLOR,
            zorder=3)
    yy2 = yy - rh
    ax.text(nx, yy2, "scrambled_labels", ha="left", va="center", fontsize=16, family=MONO, color=TEXT_COLOR,
            zorder=3)
    before_after(ax, bx, yy, SEQUENCE, SCRAMBLED_SEQUENCE)
    # Outlined tiles distinguish labels (outputs) from residues (inputs).
    before_after(ax, bx, yy2, LABELS, SCRAMBLED_LABELS, ec=SUB)

    # naive baselines
    gh3 = 2 * rh + 0.45
    gy3 = gy2 - 0.25 - gh3
    group(ax, gx, gy3, gw, gh3, "white", "baselines")
    yy = gy3 + gh3 - 0.68
    ax.text(nx, yy, "majority_class · train_mean", ha="left", va="center", fontsize=16, family=MONO,
            color=TEXT_COLOR, zorder=3)
    yy -= rh
    ax.text(nx, yy, "separation_prior", ha="left", va="center", fontsize=16, family=MONO, color=TEXT_COLOR,
            zorder=3)
    ax.text(bx, yy, "P(contact | |i − j|)", ha="left", va="center", fontsize=FS - 1,
            color=TEXT_COLOR, zorder=3)

    # ---------------------------------------------------------------- statistics
    x0 = lw + gap
    panel(ax, x0, 0, rw, H, LANE1_BG, "Statistics", "score.bootstrap_enabled")

    ax0, aw = x0 + 4.9, 4.6           # axis start and width (X_MIN → X_MAX)

    def xpos(v):
        return ax0 + (v - X_MIN) / (X_MAX - X_MIN) * aw

    bh, step = 0.6, 1.05
    top = H - 1.9
    for k, (name, v, lo, hi, stars) in enumerate(BARS):
        yy = top - k * step
        trained = k == 0
        ax.text(ax0 - 0.2, yy, name, ha="right", va="center", fontsize=14 if not trained else 16, family=MONO,
                fontweight="bold" if trained else "normal", color=TEXT_COLOR)
        # Near-zero bars are too thin for rounded corners; the CI carries them.
        if xpos(v) - xpos(0) > 0.2:
            rbox(ax, xpos(0), yy - bh / 2, xpos(v) - xpos(0), bh, MASK_BG if trained else "white", r=0.08, z=2)
        if hi > lo:
            ax.plot([xpos(lo), xpos(hi)], [yy, yy], color=TEXT_COLOR, lw=2.2, zorder=4, solid_capstyle="butt")
            for e in (lo, hi):
                ax.plot([xpos(e)] * 2, [yy - 0.15, yy + 0.15], color=TEXT_COLOR, lw=2.2, zorder=4)
        else:
            ax.plot([xpos(v)], [yy], "o", color=TEXT_COLOR, ms=6, zorder=4)
        if stars:
            ax.text(xpos(hi) + 0.15, yy - 0.05, stars, ha="left", va="center", fontsize=17,
                    fontweight="bold", color=MASK_BG, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.05", fc=LANE1_BG, ec="none"))
    base = top - (len(BARS) - 1) * step - bh / 2 - 0.15
    ax.plot([xpos(CHANCE)] * 2, [base, top + bh / 2 + 0.15], color=SUB, lw=1.5, ls="--", zorder=1)
    ax.plot([ax0, ax0], [base, top + bh / 2 + 0.15], color=SUB, lw=1.5, zorder=1)
    ax.plot([ax0, ax0 + aw], [base, base], color=SUB, lw=1.5, zorder=1)
    ax.text(xpos(CHANCE), base - 0.4, "chance", ha="center", va="center", fontsize=14, color=SUB)
    ax.text(ax0 + aw, base - 0.4, "score", ha="right", va="center", fontsize=14, color=SUB)
    ax.text(x0 + 0.5, base - 0.4, "*** q < 0.001", ha="left", va="center", fontsize=14, color=MASK_BG,
            fontweight="bold")

    sy = 2.0
    for k, head in enumerate(["bootstrap 95% CIs", "permutation p-values vs. controls", "Benjamini–Hochberg correction"]):
        ax.text(x0 + 0.5, sy - k * 0.75, head, ha="left", va="center", fontsize=FS, fontweight="bold",
                color=TEXT_COLOR)

    save(fig, out, transparent, dpi)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    p.add_argument("--opaque", action="store_true")
    p.add_argument("--dpi", type=int, default=475)
    a = p.parse_args()
    plot(a.outdir / "eval_controls.png", not a.opaque, a.dpi)


if __name__ == "__main__":
    main()
