"""Plot convergence charts from saved benchmark CSVs.

Usage:
    python plot_benchmarks.py [--reports-dir PATH] [--x-axis {time,iterations}]

Reads CSV files from _convergence_reports/ (or --reports-dir) and produces:
  - per-model per-algorithm convergence plots  (one line per trial)
  - per-model combined method comparison plots (mean lines + faint per-trial traces)
Output written alongside the CSVs.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress


matplotlib.use("Agg")  # non-interactive backend


HERE = Path(__file__).resolve().parent
DEFAULT_REPORTS_DIR = HERE / "_convergence_reports"


def plot_flow_dashboard(
    aeq_with_nodes: pd.DataFrame,
    model_name: str,
    method: str,
    rgap_target: float,
    save_path: Path,
):
    """
    Combined flow comparison dashboard for AequilibraE vs TNTP
    """
    sns.set_theme(style="whitegrid", context="paper")

    fig = plt.figure(figsize=(16, 8), dpi=150)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], wspace=0.10, hspace=0.35)
    ax_main = fig.add_subplot(gs[:, 0])

    def add_scatter(ax, x_flows, y_flows, x_label, y_label, title, marker_size, show_grid_lines):
        ax.scatter(x_flows, y_flows, alpha=0.5, s=marker_size, label="Link flows")

        x_max = float(np.max(x_flows)) * 1.02 if len(x_flows) else 1.0
        y_max = float(np.max(y_flows)) * 1.02 if len(y_flows) else 1.0
        limit = max(x_max, y_max, 1.0)

        # power = np.floor(np.log10(limit))

        # print(0, limit, power, np.ceil(limit / 10 ** power) * 10 ** power, n_ticks)
        # major_ticks = np.linspace(0, np.ceil(limit / 10 ** power) * 10 ** power, n_ticks)

        reg = linregress(x_flows, y_flows)
        x_line = np.array([0.0, limit])
        y_line = reg.intercept + reg.slope * x_line
        ax.plot(
            x_line,
            y_line,
            linestyle="--",
            linewidth=1.8,
            color="red",
            label=f"Regression  R²={reg.rvalue**2:.4f}\ny = {reg.slope:.4f}x + {reg.intercept:.4f}",
        )
        if len(x_flows) > 0 and len(y_flows) > 0:
            ax.plot(
                [0.0, limit],
                [0.0, limit],
                linestyle="-",
                color="grey",
                alpha=0.5,
                label="1:1",
            )

        ax.set_xlim(0.0, limit)
        ax.set_ylim(0.0, limit)
        ax.set_aspect("equal", adjustable="box")
        steps = [4, 6, 8]
        ax.xaxis.set_major_locator(plt.MaxNLocator(steps=steps))
        ax.yaxis.set_major_locator(plt.MaxNLocator(steps=steps))
        ax.set_anchor("W")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8", loc="upper left", fontsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color("black")

    add_scatter(
        ax_main,
        aeq_with_nodes["TNTP Solution"],
        aeq_with_nodes["PCE_AB"],
        "TNTP Reference Flow",
        "AequilibraE Flow",
        "AequilibraE vs TNTP",
        marker_size=16,
        show_grid_lines=True,
    )

    fig.suptitle(
        f"Flow Validation - {model_name}\n{method.upper()}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(save_path, dpi=plt.gcf().dpi, bbox_inches="tight")


def _load_all_reports(reports_dir: Path) -> pd.DataFrame:
    """Load and concatenate all CSVs in *reports_dir*."""
    dfs = []
    for csv_path in sorted(reports_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {reports_dir}")
    return pd.concat(dfs, ignore_index=True)


def _x_column(plot_time: bool) -> str:
    return "time" if plot_time else "index"


def _x_label(plot_time: bool) -> str:
    return "Time (s)" if plot_time else "Iterations"


def _marker_label(plot_time: bool) -> str:
    return "time" if plot_time else "iterations"


def plot_single(
    df: pd.DataFrame,
    model: str,
    algorithm: str,
    save_path: Path,
    *,
    plot_time: bool = False,
):
    """Per-model, per-algorithm convergence plot, one line per trial."""
    sns.set_theme(style="whitegrid", context="paper")
    palette = sns.color_palette()

    trials = sorted(df["trial"].unique())
    n_trials = len(trials)

    # Build x values: time or iteration index per trial
    # x_col = _x_column(plot_time)
    trial_x: list[np.ndarray] = []
    trial_rgap: list[np.ndarray] = []
    for t in trials:
        sub = df[df["trial"] == t].reset_index(drop=True)
        if plot_time:
            trial_x.append(sub["time"].values)
        else:
            trial_x.append(np.arange(len(sub), dtype=float))
        trial_rgap.append(sub["rgap"].values)

    n_pts_max = max(len(x) for x in trial_x)
    markevery = max(1, n_pts_max // 20)
    alpha = 1.0 if n_trials <= 5 else 0.4

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    for t_idx, t in enumerate(trials):
        label = f"Trial {t}" if n_trials > 1 else "AequilibraE (AoN relative gap)"
        sns.lineplot(
            x=trial_x[t_idx],
            y=trial_rgap[t_idx],
            label=label,
            ax=ax,
            marker="^",
            markevery=markevery,
            markersize=6 if n_trials > 1 else 7,
            color=palette[1],
            linewidth=1.2 if n_trials > 1 else 2,
            linestyle=":",
            alpha=alpha,
        )

    ax.set_xlim(left=0)
    ax.set_yscale("log")
    ax.grid(True, which="minor", axis="y", linewidth=0.7, alpha=0.3)
    ax.set_xlabel(_x_label(plot_time), labelpad=10)
    ax.set_ylabel("Relative Gap", labelpad=10)
    title_extra = f"  ({n_trials} trials)" if n_trials > 1 else ""
    ax.set_title(
        f"Convergence - {model}\n{algorithm.upper()}{title_extra}",
        pad=14,
        fontweight="bold",
        fontsize=14,
    )
    ax.text(
        0.02,
        0.03,
        f"Markers every {markevery} {_marker_label(plot_time)} ticks",
        transform=ax.transAxes,
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "0.9",
        },
    )
    ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8")
    if not plot_time:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_method_comparison(
    df: pd.DataFrame,
    model: str,
    save_path: Path,
    *,
    plot_time: bool = False,
):
    """Per-model comparison of all algorithms. Mean lines in legend, faint per-trial traces behind."""
    sns.set_theme(style="whitegrid", context="paper")
    palette = sns.color_palette()

    algorithms = sorted(df["algorithm"].unique(), key=str.upper)

    # Collect per-algorithm trial data and compute means
    alg_trials: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    alg_means: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for alg in algorithms:
        sub = df[df["algorithm"] == alg]
        trials = sorted(sub["trial"].unique())
        pairs = []
        for t in trials:
            tdf = sub[sub["trial"] == t].reset_index(drop=True)
            if plot_time:
                x = tdf["time"].values
            else:
                x = np.arange(len(tdf), dtype=float)
            pairs.append((x, tdf["rgap"].values))
        alg_trials[alg] = pairs

        # Mean: align by iteration index regardless of x-axis mode
        tdf_idx = sub.copy()
        tdf_idx["iter_trial"] = tdf_idx.groupby("trial").cumcount()
        mean_rgap = tdf_idx.groupby("iter_trial")["rgap"].mean()
        if plot_time:
            mean_time = tdf_idx.groupby("iter_trial")["time"].mean()
            alg_means[alg] = (mean_time.values, mean_rgap.values)
        else:
            alg_means[alg] = (np.arange(len(mean_rgap), dtype=float), mean_rgap.values)

    max_len = max(len(v[0]) for v in alg_means.values()) if alg_means else 1
    markevery = max(1, max_len // 20)
    n_trials_total = sum(len(v) for v in alg_trials.values())

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # faint per-trial traces (no legend), only when >1 trial exists
    if n_trials_total > len(algorithms):
        for i, alg in enumerate(algorithms):
            color = palette[i % len(palette)]
            for x, y in alg_trials[alg]:
                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=1.5,
                    alpha=0.5,
                )

    for i, alg in enumerate(algorithms):
        x_mean, y_mean = alg_means[alg]
        sns.lineplot(
            x=x_mean,
            y=y_mean,
            label=alg.upper(),
            ax=ax,
            marker="^",
            markevery=markevery,
            markersize=6,
            color=palette[i % len(palette)],
            linewidth=2,
        )

    ax.set_xlim(left=0)
    ax.set_yscale("log")
    ax.grid(True, which="minor", axis="y", linewidth=0.7, alpha=0.3)
    ax.set_xlabel(_x_label(plot_time), labelpad=10)
    ax.set_ylabel("Relative Gap", labelpad=10)
    ax.set_title(
        f"Convergence - {model}\nAll Methods",
        pad=14,
        fontweight="bold",
        fontsize=14,
    )
    ax.text(
        0.02,
        0.03,
        f"Markers every {markevery} {_marker_label(plot_time)} ticks",
        transform=ax.transAxes,
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "0.9",
        },
    )
    ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8")
    if not plot_time:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_convergence(reports_dir: Path, plot_time: bool):
    all_data = _load_all_reports(reports_dir)
    out_dir = reports_dir  # write plots alongside CSVs

    mode = "time" if plot_time else "iters"
    models = sorted(all_data["model"].unique())

    for model in models:
        model_df = all_data[all_data["model"] == model]
        algorithms = sorted(model_df["algorithm"].unique(), key=str.upper)

        for alg in algorithms:
            sub = model_df[model_df["algorithm"] == alg].reset_index(drop=True)
            plot_single(
                sub,
                model,
                alg,
                out_dir / f"{model}_{alg}_convergence_{mode}.png",
                plot_time=plot_time,
            )

        plot_method_comparison(
            model_df.reset_index(drop=True),
            model,
            out_dir / f"{model}_all_methods_convergence_{mode}.png",
            plot_time=plot_time,
        )

    print(f"Plots written to {out_dir}")


def compare_flows_with_tntp(reports_dir: Path):
    out_dir = reports_dir

    model = "Anaheim"
    alg = "bfw"

    results_with_nodes = pd.read_parquet(reports_dir / f"{model}_{alg}_results_with_nodes.parquet")
    plot_flow_dashboard(results_with_nodes, model, alg, 0, out_dir / f"{model}_{alg}_flow_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate convergence plots from benchmark CSV reports and/or compare flow to benchmark results."
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help=f"Directory containing CSV report files (default: {DEFAULT_REPORTS_DIR})",
    )
    parser.add_argument(
        "--convergence",
        action="store_true",
        default=False,
        help="Make convergence plots (default: False)",
    )
    parser.add_argument(
        "--x-axis",
        choices=["time", "iterations"],
        default="iterations",
        help="X-axis mode: 'time' (wall clock seconds) or 'iterations' (default)",
    )
    parser.add_argument(
        "--compare-flow",
        action="store_true",
        default=False,
        help="Make flow comparison plots (default: False)",
    )
    args = parser.parse_args()
    if args.convergence:
        print("Plotting convergence...")
        plot_all_convergence(args.reports_dir, plot_time=(args.x_axis == "time"))
    if args.compare_flow:
        print("Comparing flow...")
        compare_flows_with_tntp(args.reports_dir)
