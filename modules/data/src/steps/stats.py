import json
import logging
import time
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless: figures are only ever saved to disk
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import polars as pl
from pydantic import BaseModel, ConfigDict

from modules.data.src.dataset.dataset import Dataset
from modules.data.src.utils.io_utils import (
    override_or_default,
    resolve_parquet_source,
    resolve_path,
)

logger = logging.getLogger(__name__)

_ACCENT = "#3b7dd8"
_ACCENT_DARK = "#1f4e8c"


def _style_axes(ax: Axes) -> None:
    """Shared minimal styling: light grid, no top/right border."""
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


class StatsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    override_source: str | None = (
        None  # Explicit source (file or directory); defaults to the assembled train output.
    )
    override_output: str | None = (
        None  # Explicit output path (file or directory); defaults under dataset.train_path.
    )
    length_col: str = "sequence_length"
    score_col: str = "red_score"
    sequence_col: str = "sequence"
    quantiles: List[float] = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    count_ambiguous: bool = True
    ambiguous_chars: str = "XBZJ"
    # Counts distinct clusters per `cluster_rep_at_*` column, if present.
    count_cluster_thresholds: bool = True
    cluster_col_prefix: str = "cluster_rep_at_"
    # Renders distribution figures next to the JSON summary.
    make_plots: bool = True
    plot_num_bins: int = 50
    plot_dpi: int = 150


class StatsStep:
    def __init__(self, config: StatsConfig) -> None:
        self.config = config

    def run(self, dataset: Dataset | None = None) -> None:
        if dataset is not None:
            dataset.setup_directories()

        if dataset is None and self.config.override_source is None:
            raise ValueError(
                "StatsConfig requires override_source unless run(dataset) is used."
            )
        if dataset is None and self.config.override_output is None:
            raise ValueError(
                "StatsConfig requires override_output unless run(dataset) is used."
            )

        if self.config.override_source is not None:
            source = resolve_path(
                self.config.override_source, Path(), "Stats source", required=True
            )
            assert source is not None  # required=True guarantees a path or raise
        else:
            # assemble step may write a single file or a shard directory.
            assert dataset is not None
            assembled_dir = dataset.train_path / f"{dataset.name}_assembled"
            assembled_file = dataset.train_path / f"{dataset.name}_assembled.parquet"
            if assembled_dir.is_dir() and any(assembled_dir.glob("*.parquet")):
                source = assembled_dir
            elif assembled_file.exists():
                source = assembled_file
            else:
                raise FileNotFoundError(
                    "Could not find assembled parquet output for stats. "
                    "Run assemble step first or set steps.stats.override_source."
                )

        output_default = (
            dataset.stats_path / f"{dataset.name}_stats_summary.json"
            if dataset
            else Path()
        )
        output_path = override_or_default(self.config.override_output, output_default)
        if output_path.is_dir():
            output_path = output_path / "stats_summary.json"
        logger.info("Computing stats from %s", source)

        t0 = time.time()
        stats = self.compute_stats(
            source,
            self.config.length_col,
            self.config.score_col,
            self.config.quantiles,
            sequence_col=self.config.sequence_col,
            count_ambiguous=self.config.count_ambiguous,
            ambiguous_chars=self.config.ambiguous_chars,
            count_cluster_thresholds=self.config.count_cluster_thresholds,
            cluster_col_prefix=self.config.cluster_col_prefix,
        )
        elapsed = time.time() - t0
        stats["elapsed_seconds"] = round(elapsed, 3)

        logger.info(json.dumps(stats, indent=2, default=str))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(stats, indent=2, default=str))
        logger.info("Wrote stats to %s", output_path)

        if self.config.make_plots:
            plots_dir = output_path.parent
            name = dataset.name if dataset else Path(source).stem
            self.make_plots(
                source,
                plots_dir,
                name,
                self.config.length_col,
                self.config.score_col,
                self.config.cluster_col_prefix,
                stats=stats,
                num_bins=self.config.plot_num_bins,
                dpi=self.config.plot_dpi,
            )
            logger.info("Wrote distribution figures to %s", plots_dir)

    @staticmethod
    def compute_stats(
        source: str | Path,
        length_col: str,
        score_col: str,
        quantiles: List[float],
        sequence_col: str = "sequence",
        count_ambiguous: bool = True,
        ambiguous_chars: str = "XBZJ",
        count_cluster_thresholds: bool = True,
        cluster_col_prefix: str = "cluster_rep_at_",
    ) -> dict:
        # resolve_parquet_source handles both a single file and a shard directory.
        raw_lf = resolve_parquet_source(source)

        # Box plots need q1/median/q3, so always include them.
        quantiles = sorted(set(quantiles) | {0.25, 0.5, 0.75})

        cluster_cols: List[str] = []
        if count_cluster_thresholds:
            cluster_cols = sorted(
                c
                for c in raw_lf.collect_schema().names()
                if c.startswith(cluster_col_prefix)
            )
            logger.info(
                "Detected %d cluster threshold column(s): %s",
                len(cluster_cols),
                cluster_cols,
            )

        # Each block runs as its own narrowly-scoped query so peak memory stays
        # bounded to one pass at a time (numeric agg, string scan, cluster
        # hash sets), instead of holding all of them at once.
        numeric_exprs = [
            pl.len().alias("num_rows"),
            pl.col(length_col).min().alias("length_min"),
            pl.col(length_col).max().alias("length_max"),
            pl.col(length_col).mean().alias("length_mean"),
            pl.col(length_col).median().alias("length_median"),
            pl.col(length_col).std().alias("length_std"),
            pl.col(length_col).sum().alias("sum_len"),
            pl.col(score_col).mean().alias("score_mean"),
            pl.col(score_col).median().alias("score_median"),
            pl.col(score_col).std().alias("score_std"),
            pl.col(score_col).var().alias("score_var"),
            pl.col(score_col).min().alias("score_min"),
            pl.col(score_col).max().alias("score_max"),
        ]
        for q in quantiles:
            numeric_exprs.append(pl.col(length_col).quantile(q).alias(f"length_q{q}"))
            numeric_exprs.append(pl.col(score_col).quantile(q).alias(f"score_q{q}"))

        logger.info("Running numeric aggregation pass (length + score columns)...")
        t_main = time.time()
        result = (
            raw_lf.select([length_col, score_col])
            .select(numeric_exprs)
            .collect(engine="streaming")
        )
        stats = result.row(0, named=True)
        logger.info(
            "Numeric aggregation complete in %.2fs (%s rows scanned).",
            time.time() - t_main,
            stats.get("num_rows"),
        )

        if count_ambiguous:
            # Own pass: a full string scan should not inflate the memory
            # footprint of the numeric aggregation above.
            logger.info("Running ambiguous-residue pass over '%s'...", sequence_col)
            t_amb = time.time()
            ambiguous_pattern = f"[{ambiguous_chars}]"
            ambiguous_count_expr = pl.col(sequence_col).str.count_matches(
                ambiguous_pattern
            )
            amb_row = (
                raw_lf.select(sequence_col)
                .select(
                    ambiguous_count_expr.sum().alias("ambiguous_count"),
                    (ambiguous_count_expr > 0).sum().alias("sequences_with_ambiguous"),
                )
                .collect(engine="streaming")
                .row(0, named=True)
            )
            stats.update(amb_row)
            sum_len = stats.get("sum_len") or 0
            stats["ambiguous_fraction"] = (
                stats["ambiguous_count"] / sum_len if sum_len else None
            )
            logger.info(
                "Ambiguous residues: %s (%.4f%% of %s total residues) in %.2fs.",
                stats["ambiguous_count"],
                (stats["ambiguous_fraction"] or 0) * 100,
                sum_len,
                time.time() - t_amb,
            )
        stats["ambiguous_residues_computed"] = count_ambiguous

        if cluster_cols:
            # approx_n_unique (HyperLogLog) instead of exact n_unique, which
            # would build a full hash set over every distinct cluster ID.
            logger.info(
                "Running cluster threshold pass(es) over %d column(s)...",
                len(cluster_cols),
            )
            t_clusters = time.time()
            for col in cluster_cols:
                num_clusters = (
                    raw_lf.select(col)
                    .select(pl.col(col).approx_n_unique().alias("n"))
                    .collect(engine="streaming")
                    .item()
                )
                stats[f"num_clusters_{col}"] = num_clusters
            num_rows = stats.get("num_rows") or 0
            for col in cluster_cols:
                stats[f"cluster_reduction_{col}"] = (
                    stats[f"num_clusters_{col}"] / num_rows if num_rows else None
                )
            logger.info(
                "Cluster threshold counts (approx.): %s (%.2fs).",
                {col: stats[f"num_clusters_{col}"] for col in cluster_cols},
                time.time() - t_clusters,
            )
        stats["cluster_thresholds_computed"] = bool(cluster_cols)

        # Mode requires a value_counts pass; kept separate for the same
        # memory reasons as above.
        logger.info("Computing score mode (separate group_by pass)...")
        t_mode = time.time()
        mode_row = (
            raw_lf.select(score_col)
            .group_by(score_col)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .limit(1)
            .collect(engine="streaming")
        )
        logger.info("Score mode pass complete in %.2fs.", time.time() - t_mode)
        if mode_row.height:
            stats["score_mode"] = mode_row[score_col][0]
            stats["score_mode_count"] = mode_row["count"][0]
        else:
            stats["score_mode"] = None
            stats["score_mode_count"] = 0

        return stats

    @staticmethod
    def make_plots(
        source: str | Path,
        output_dir: Path,
        name: str,
        length_col: str,
        score_col: str,
        cluster_col_prefix: str,
        stats: dict,
        num_bins: int = 50,
        dpi: int = 150,
    ) -> None:
        StatsStep.plot_distribution(
            source,
            length_col,
            output_dir / f"{name}_length_dist.png",
            title=f"{name}: sequence length distribution",
            xlabel="Sequence length",
            num_bins=num_bins,
            dpi=dpi,
        )
        logger.info("Plot 1/5 done: length distribution.")
        StatsStep.plot_distribution(
            source,
            score_col,
            output_dir / f"{name}_score_dist.png",
            title=f"{name}: score distribution",
            xlabel=score_col,
            num_bins=num_bins,
            dpi=dpi,
        )
        logger.info("Plot 2/5 done: score distribution.")
        StatsStep.plot_cluster_reduction(
            stats,
            cluster_col_prefix,
            output_dir / f"{name}_cluster_reduction.png",
            title=f"{name}: cluster count by identity threshold",
            dpi=dpi,
        )
        logger.info("Plot 3/5 done: cluster reduction.")
        StatsStep.plot_boxplot(
            stats,
            output_dir / f"{name}_length_boxplot.png",
            title=f"{name}: sequence length box plot",
            ylabel="Sequence length",
            stat_prefix="length",
            dpi=dpi,
        )
        logger.info("Plot 4/5 done: length box plot.")
        StatsStep.plot_boxplot(
            stats,
            output_dir / f"{name}_score_boxplot.png",
            title=f"{name}: score box plot",
            ylabel=score_col,
            stat_prefix="score",
            dpi=dpi,
        )
        logger.info("Plot 5/5 done: score box plot.")

    @staticmethod
    def plot_distribution(
        source: str | Path,
        column: str,
        output_path: Path,
        title: str,
        xlabel: str,
        num_bins: int = 50,
        dpi: int = 150,
    ) -> None:
        # Bin counts are computed streaming in Polars; only the small
        # resulting histogram is pulled into Python for plotting.
        lf = resolve_parquet_source(source).select(column)
        bounds = lf.select(
            pl.col(column).min().alias("min"), pl.col(column).max().alias("max")
        ).collect(engine="streaming")
        col_min, col_max = bounds["min"][0], bounds["max"][0]
        if col_min is None or col_max is None:
            logger.warning("No data for column %s; skipping %s", column, output_path)
            return
        if col_min == col_max:
            col_max = col_min + 1  # avoid a zero-width histogram range

        hist = (
            lf.select(
                pl.col(column).hist(
                    bin_count=num_bins, include_category=False, include_breakpoint=True
                )
            )
            .unnest(column)
            .collect(engine="streaming")
        )
        breakpoints = hist["breakpoint"].to_list()
        counts = hist["count"].to_list()
        width = (col_max - col_min) / num_bins

        # Always render both a linear and a log-y version, since a dominant
        # peak can hide smaller bins on a linear y-axis.
        for log_y, path in (
            (False, output_path),
            (True, output_path.with_stem(output_path.stem + "_log")),
        ):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(
                breakpoints,
                counts,
                width=width,
                color=_ACCENT,
                edgecolor=_ACCENT_DARK,
                linewidth=0.5,
                alpha=0.75,
            )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            _style_axes(ax)
            if log_y:
                ax.set_yscale("log")
            fig.tight_layout()
            fig.savefig(path, dpi=dpi)
            plt.close(fig)

    @staticmethod
    def plot_cluster_reduction(
        stats: dict,
        cluster_col_prefix: str,
        output_path: Path,
        title: str,
        dpi: int = 150,
    ) -> None:
        # Reuses num_clusters_* from compute_stats instead of re-scanning.
        prefix = f"num_clusters_{cluster_col_prefix}"
        keys = sorted(k for k in stats if k.startswith(prefix))
        if not keys:
            logger.warning(
                "No %s* stats found; skipping cluster reduction plot.",
                prefix,
            )
            return

        thresholds = [k[len(prefix) :] for k in keys]
        num_clusters = [stats[k] for k in keys]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(
            thresholds,
            num_clusters,
            color=_ACCENT,
            edgecolor=_ACCENT_DARK,
            linewidth=0.5,
            alpha=0.75,
        )
        ax.set_title(title)
        ax.set_xlabel("Identity threshold (%)")
        ax.set_ylabel("Number of clusters")
        ax.axhline(
            stats.get("num_rows", 0),
            color="gray",
            linestyle="--",
            label="Total sequences",
        )
        ax.legend()
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)

    @staticmethod
    def _tukey_box_stats(
        label: str, q1: float, med: float, q3: float, lo: float, hi: float
    ) -> dict:
        # Standard Tukey whiskers (1.5*IQR), clipped to min/max since we
        # don't have raw values to compute real fliers from aggregates.
        iqr = q3 - q1
        whislo = max(lo, q1 - 1.5 * iqr)
        whishi = min(hi, q3 + 1.5 * iqr)
        return {
            "label": label,
            "whislo": whislo,
            "q1": q1,
            "med": med,
            "q3": q3,
            "whishi": whishi,
            "fliers": [],
        }

    @staticmethod
    def plot_boxplot(
        stats: dict,
        output_path: Path,
        title: str,
        ylabel: str,
        stat_prefix: str,
        dpi: int = 150,
    ) -> None:
        q1 = stats.get(f"{stat_prefix}_q0.25")
        med = stats.get(f"{stat_prefix}_q0.5")
        q3 = stats.get(f"{stat_prefix}_q0.75")
        lo = stats.get(f"{stat_prefix}_min")
        hi = stats.get(f"{stat_prefix}_max")
        if q1 is None or med is None or q3 is None or lo is None or hi is None:
            logger.warning("Missing %s stats; skipping %s", stat_prefix, output_path)
            return

        box_stats = [
            StatsStep._tukey_box_stats(
                ylabel, float(q1), float(med), float(q3), float(lo), float(hi)
            )
        ]

        fig, ax = plt.subplots(figsize=(4, 6))
        ax.bxp(
            box_stats,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": _ACCENT, "edgecolor": _ACCENT_DARK, "alpha": 0.75},
            medianprops={"color": "black"},
            whiskerprops={"color": _ACCENT_DARK},
            capprops={"color": _ACCENT_DARK},
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
