"""Main code components of modules/evaluate (same layout as plot_pretrain_components.py).

    python figures/plot_eval_components.py -o figures/eval_components.png
"""

import argparse
from pathlib import Path

from plot_data_pipeline import GREEN, GREY, LANE1_BG, LANE2_BG, new_axes, rbox, save
from plot_masked_sequence import MASK_BG, RESIDUE_BG, TEXT_COLOR
from plot_pretrain_components import MONO, NEUTRAL_BG, arrow, card

GROUPS = [
    ("Pipeline steps", LANE2_BG, GREEN, TEXT_COLOR, [
        ("prepare.py", "tokenize once · load model · cache"),
        ("tune.py", "probe / fine-tune · HPO"),
        ("predict.py", "model + control predictions"),
        ("score.py", "metrics · CIs · vs. controls"),
    ]),
    ("Tasks & heads", LANE1_BG, MASK_BG, "white", [
        ("tasks/registry.py", "@register_task plug-ins"),
        ("tasks/*.py", "7 task handlers"),
        ("heads.py · contact_model.py", "6 probe heads · bilinear"),
    ]),
    ("Controls & stats", NEUTRAL_BG, RESIDUE_BG, TEXT_COLOR, [
        ("controls.py", "random · scrambled · baselines"),
        ("bootstrap.py", "CIs · permutation · BH"),
        ("tracking.py", "metrics → W&B"),
    ]),
]
TOP = [
    ("scripts/", "results tables · sourcing", NEUTRAL_BG, TEXT_COLOR),
    ("run_evaluate.py", "main entry point", TEXT_COLOR, "white"),
    ("configs/ + config.py", "YAML → Pydantic config", NEUTRAL_BG, TEXT_COLOR),
]
STANDALONE = [
    ("proteingym/", "run_proteingym_*.py", "zero-shot & supervised\nmutation-effect benchmark"),
    ("dataset_sourcing/", "scripts/dataset_sourcing/", "download → split → card\n→ HF Hub"),
]


def plot_components(out: Path, transparent: bool, dpi: int):
    gw, ggap, cw, ch, cgap = 7.8, 0.7, 7.0, 1.5, 0.35
    span = 3 * ch + 2 * cgap                # vertical span of the cards in a group
    gx = gw + ggap                          # main groups start right of the standalone column
    width = gx + 3 * gw + 2 * ggap
    gy = 0.9                                # above the abbreviation legend
    gh = 1.1 + span + 0.4
    top_y, top_h = gy + gh + 1.6, 1.5
    fig, ax = new_axes(-0.3, width + 0.3, -0.6, top_y + top_h + 0.3)

    centers = []
    for g, (title, bg, fc, tc, files) in enumerate(GROUPS):
        x = gx + g * (gw + ggap)
        centers.append(x + gw / 2)
        rbox(ax, x, gy, gw, gh, bg, r=0.3, z=0)
        ax.text(x + gw / 2, gy + gh - 0.6, title, ha="center", va="center", fontsize=22, fontweight="bold",
                color=TEXT_COLOR)
        # Fit n cards in the same vertical span as three standard cards.
        n = len(files)
        g2 = cgap if n == 3 else 0.25
        h2 = (span - (n - 1) * g2) / n
        for k, (name, desc) in enumerate(files):
            y = gy + gh - 1.1 - (k + 1) * h2 - k * g2
            card(ax, x + (gw - cw) / 2, y, cw, h2, fc, tc, name, desc)

    for k, (name, desc, fc, tc) in enumerate(TOP):
        card(ax, centers[k] - cw / 2, top_y, cw, top_h, fc, tc, name, desc)
    arrow(ax, (centers[0] + cw / 2 + 0.05, top_y + top_h / 2), (centers[1] - cw / 2 - 0.05, top_y + top_h / 2))
    arrow(ax, (centers[2] - cw / 2 - 0.05, top_y + top_h / 2), (centers[1] + cw / 2 + 0.05, top_y + top_h / 2))
    xs, y0, y1 = centers[1], top_y - 0.05, gy + gh + 0.05
    ym = (y0 + y1) / 2
    ax.plot([xs, xs], [y0, ym], color=TEXT_COLOR, lw=2.2)
    ax.plot([centers[0], centers[2]], [ym, ym], color=TEXT_COLOR, lw=2.2)
    for c in centers:
        arrow(ax, (c, ym), (c, y1))

    # Standalone pipelines: left column, own entry points.
    rbox(ax, 0, gy, gw, gh, "none", ec=GREY, ls="--", lw=1.5, r=0.3, z=0)
    ax.text(gw / 2, gy + gh - 0.6, "Standalone", ha="center", va="center", fontsize=22, fontweight="bold",
            color=TEXT_COLOR)
    sh = (span - cgap) / 2
    for k, (name, entry, desc) in enumerate(STANDALONE):
        x, y = (gw - cw) / 2, gy + gh - 1.1 - (k + 1) * sh - k * cgap
        rbox(ax, x, y, cw, sh, NEUTRAL_BG)
        ax.text(x + cw / 2, y + sh - 0.45, name, ha="center", va="center", fontsize=16, fontweight="bold",
                color=TEXT_COLOR, family=MONO)
        ax.text(x + cw / 2, y + sh - 0.95, entry, ha="center", va="center", fontsize=14, color=GREY, family=MONO)
        ax.text(x + cw / 2, y + 0.75, desc, ha="center", va="center", fontsize=15, color=TEXT_COLOR,
                linespacing=1.3)

    ax.text(width / 2, -0.05, "HPO = hyperparameter optimization   ·   CIs = bootstrap confidence intervals   ·   "
            "BH = Benjamini–Hochberg multiple-testing correction", ha="center", va="center", fontsize=18,
            color=GREY)

    save(fig, out, transparent, dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--opaque", action="store_true")
    parser.add_argument("--dpi", type=int, default=380)  # wide figure: keeps text crisp, < 25 MP
    parser.add_argument("-o", "--out", type=Path, default=Path(__file__).with_name("eval_components.png"))
    args = parser.parse_args()
    plot_components(args.out, not args.opaque, args.dpi)


if __name__ == "__main__":
    main()
