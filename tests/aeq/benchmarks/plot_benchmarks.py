"""Plot convergence charts from saved benchmark CSVs.

Usage:
    python plot_benchmarks.py [--convergence] [--compare-flow] [--reports-dir PATH] [--x-axis {time,iterations}]

If --convergence is specified, reads CSV files from _convergence_reports/ (or --reports-dir) and produces:
  - per-model per-algorithm convergence plots  (one line per trial)
  - per-model combined method comparison plots (mean lines + faint per-trial traces)
If --compare-flow is specified, reads parquet files from _convergence_reports/ (or --reports-dir) and produces:
  - per-model per-algorithm plots that compare the found flows between nodes to known solutions
Output written alongside the CSVs and parquet files.

"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress


HERE = Path(__file__).resolve().parent
DEFAULT_REPORTS_DIR = HERE / "_convergence_reports"


def plot_flow_dashboard(
    aeq_with_nodes: pd.DataFrame,
    model_name: str,
    method: str,
    heap: str,
    rgap_target: float,
    save_path: Path,
):
    """
    Flow comparison dashboard for AequilibraE vs TNTP
    """
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    x_flows = aeq_with_nodes["TNTP Solution"]
    y_flows = aeq_with_nodes["PCE_AB"]

    ax.scatter(x_flows, y_flows, alpha=0.5, s=16, label="Link flows")

    x_max = float(np.max(x_flows)) * 1.02 if len(x_flows) else 1.0
    y_max = float(np.max(y_flows)) * 1.02 if len(y_flows) else 1.0
    limit = max(x_max, y_max, 1.0)

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
    ax.plot([0.0, limit], [0.0, limit], linestyle="-", color="grey", alpha=0.5, label="1:1")

    ax.set_xlim(0.0, limit)
    ax.set_ylim(0.0, limit)
    ax.set_aspect("equal", adjustable="box")

    steps = [4, 6, 8]
    ax.xaxis.set_major_locator(plt.MaxNLocator(steps=steps))
    ax.yaxis.set_major_locator(plt.MaxNLocator(steps=steps))

    ax.set_xlabel("TNTP Reference Flow")
    ax.set_ylabel("AequilibraE Flow")
    ax.set_title(
        f"Flow Validation for {model_name} with heap {heap} using {method.upper()}",
        fontweight="bold",
        fontsize=11,
    )
    ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8", loc="upper left", fontsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")

    plt.savefig(save_path, dpi=fig.dpi, bbox_inches="tight")
    plt.close(fig)


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
    heap: str,
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
        f"Convergence - {model}, heap {heap}\n{algorithm.upper()}{title_extra}",
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
    heap: str,
    save_path: Path,
    *,
    plot_time: bool = False,
    plot_all_trials: bool = False,
):
    """Per-model comparison of all algorithms. Mean lines in legend. If the x axis is time, it will also plot the
    slowest and fastest trials. If the x axis is iterations, it will also plot the worst performing and best performing
    trails based on the relative gap."""
    sns.set_theme(style="whitegrid", context="paper")
    palette = sns.color_palette()

    algorithms = sorted(df["algorithm"].unique(), key=str.upper)

    # Collect per-algorithm trial data and compute means
    alg_trials: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    alg_means: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    alg_best_trial: dict[str, int] = {}
    alg_worst_trial: dict[str, int] = {}

    for alg in algorithms:
        best_trial_quantity = float("inf")  # worst case for iter is large gap, worst case for time is large time
        worst_trial_quantity = -1  # best case for iter is small gap, best case for time is small time
        sub = df[df["algorithm"] == alg]
        trials = sorted(sub["trial"].unique())
        x_y_pairs = []
        for i, t in enumerate(trials):
            tdf = sub[sub["trial"] == t].reset_index(drop=True)
            if plot_time:
                x = tdf["time"].values
            else:
                x = np.arange(len(tdf), dtype=float)
            y = tdf["rgap"].values
            x_y_pairs.append((x, y))
            if plot_time:
                # compare times for best and worst trial
                if x[-1] < best_trial_quantity:
                    alg_best_trial[alg] = i
                    best_trial_quantity = x[-1]
                if x[-1] > worst_trial_quantity:
                    alg_worst_trial[alg] = i
                    worst_trial_quantity = x[-1]
            else:
                # compare gaps for best and worst trial
                if y[-1] < best_trial_quantity:
                    alg_best_trial[alg] = i
                    best_trial_quantity = y[-1]
                if y[-1] > worst_trial_quantity:
                    alg_worst_trial[alg] = i
                    worst_trial_quantity = y[-1]

        alg_trials[alg] = x_y_pairs

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
        if plot_all_trials:
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
        else:
            for i, alg in enumerate(algorithms):
                color = palette[i % len(palette)]
                best_x, best_y = alg_trials[alg][alg_best_trial[alg]]
                worst_x, worst_y = alg_trials[alg][alg_worst_trial[alg]]
                ax.plot(
                    best_x,
                    best_y,
                    color=color,
                    linewidth=1.5,
                    alpha=0.8,
                    linestyle="dashed",
                    label=f"{alg.upper()} best trial: {alg_best_trial[alg]}",
                )
                ax.plot(
                    worst_x,
                    worst_y,
                    color=color,
                    linewidth=1.5,
                    alpha=0.8,
                    linestyle="dotted",
                    label=f"{alg.upper()} worst trial: {alg_worst_trial[alg]}",
                )

    for i, alg in enumerate(algorithms):
        x_mean, y_mean = alg_means[alg]
        sns.lineplot(
            x=x_mean,
            y=y_mean,
            label=f"{alg.upper()} mean",
            ax=ax,
            marker="^",
            markevery=markevery,
            markersize=6,
            color=palette[i % len(palette)],
            linewidth=1,
            alpha=0.8,
        )

    ax.set_xlim(left=0)
    ax.set_yscale("log")
    ax.grid(True, which="minor", axis="y", linewidth=0.7, alpha=0.3)
    ax.set_xlabel(_x_label(plot_time), labelpad=10)
    ax.set_ylabel("Relative Gap", labelpad=10)
    ax.set_title(
        f"Convergence - {model}, heap {heap}\nAll Methods",
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


def make_all_convergence_plots(reports_dir: Path, plot_time: bool):
    print("Making convergence plots...")

    all_data = _load_all_reports(reports_dir)
    out_dir = reports_dir  # write plots alongside CSVs

    mode = "time" if plot_time else "iters"
    models = sorted(all_data["model"].unique())

    for model in models:
        model_df = all_data[all_data["model"] == model]
        algorithms = sorted(model_df["algorithm"].unique(), key=str.upper)
        heaps = sorted(model_df["heap"].unique(), key=str.upper)
        for heap in heaps:

            for alg in algorithms:
                sub = model_df[(model_df["algorithm"] == alg) & (model_df["heap"] == heap)].reset_index(drop=True)
                plot_single(
                    sub,
                    model,
                    alg,
                    heap,
                    out_dir / f"{model}_{alg}_{heap}_convergence_{mode}.png",
                    plot_time=plot_time,
                )

            plot_method_comparison(
                model_df[(model_df["heap"] == heap)].reset_index(drop=True),
                model,
                heap,
                out_dir / f"{model}_{heap}_all_methods_convergence_{mode}.png",
                plot_time=plot_time,
            )
    print(f"    Convergence plots written to {out_dir}")


def make_all_flow_comparison_plots(reports_dir: Path):
    print("Making flow comparison plots...")

    out_dir = reports_dir
    parquet_files = sorted(reports_dir.glob("*_results_with_nodes.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {reports_dir}")
        return
    for parquet_path in parquet_files:
        # Parse model and algorithm from filename: {model}_{alg}_results_with_nodes.parquet
        stem = parquet_path.stem.replace("_results_with_nodes", "")
        model, alg, heap = stem.rsplit("_", 2)
        results_with_nodes = pd.read_parquet(parquet_path)
        plot_flow_dashboard(results_with_nodes, model, alg, heap, 0, out_dir / f"{model}_{alg}_{heap}_flow_comparison.png")
    print(f"    Flow comparison plots written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate convergence plots from benchmark CSV reports and/or compare flows to benchmark results."
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=(os.environ.get("BENCHMARK_REPORTS_DIR") or DEFAULT_REPORTS_DIR),
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
        make_all_convergence_plots(args.reports_dir, plot_time=(args.x_axis == "time"))
    if args.compare_flow:
        make_all_flow_comparison_plots(args.reports_dir)
