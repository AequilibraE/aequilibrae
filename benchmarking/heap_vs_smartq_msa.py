"""Benchmark MSA traffic assignment with the 4-ary heap vs the smart queue.

The backend is selected by the AEQ_SMARTQ environment variable, which is
read at import time inside aequilibrae.paths.cython.AoN. We therefore run
each backend in a fresh Python subprocess.

Usage (from the repo root):

    python benchmarking/heap_vs_smartq_msa.py --model-path D:\\tmp\\Chicago_aeq

Optional flags: --iterations, --cores, --matrix-name, --rebuild, --smoke,
--output-dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_matrix_name(model_path: Path, mode: str, requested: Optional[str]) -> str:
    from aequilibrae import Project

    warnings.filterwarnings("ignore")
    project = Project.from_path(model_path)
    try:
        project.network.build_graphs(modes=[mode])
        names = [str(v) for v in project.matrices.list()["name"].tolist()]
    finally:
        project.close()

    if not names:
        raise RuntimeError(f"No matrices found in {model_path}")
    if requested:
        if requested not in names:
            raise RuntimeError(f"Matrix {requested!r} not found. Available: {names}")
        return requested
    if len(names) == 1:
        return names[0]
    for candidate in ("demand", "demand_omx"):
        if candidate in names:
            return candidate
    return names[0]


def _run_single_backend(
    backend: str,
    model_path: Path,
    mode: str,
    matrix_name: str,
    iterations: int,
    cores: int,
    rgap_target: float,
    smoke: bool,
    algorithm: str,
    repetitions: int,
) -> Dict:
    """Open the project, configure assignment, time execute() over N reps.

    The project / graph / matrix / TrafficClass / TrafficAssignment are
    all constructed once and reused across repetitions, so each repeated
    timing is just the cost of a fresh `assignment.execute()` (which
    internally resets per-iteration state).
    """
    from time import perf_counter

    from aequilibrae import Project, TrafficAssignment, TrafficClass

    warnings.filterwarnings("ignore")

    project = Project.from_path(model_path)
    try:
        project.network.build_graphs(modes=[mode])
        graph = project.network.graphs[mode]
        graph.set_blocked_centroid_flows(False)
        graph.set_graph("free_flow_time")

        matrix = project.matrices.get_matrix(matrix_name)
        try:
            matrix.computational_view()

            elapsed_list: List[float] = []
            iters_run_list: List[int] = []
            rgap_list: List[float] = []
            link_load_sums: List[float] = []
            link_load_maxs: List[float] = []
            cores_used = 0

            for _ in range(max(1, repetitions)):
                assigclass = TrafficClass("car", graph, matrix)

                assignment = TrafficAssignment(project)
                assignment.add_class(assigclass)
                assignment.set_vdf("BPR")
                assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
                assignment.set_capacity_field("capacity")
                assignment.set_time_field("free_flow_time")
                assignment.set_algorithm(algorithm)
                assignment.max_iter = iterations
                assignment.rgap_target = rgap_target
                if cores > 0:
                    assignment.set_cores(cores)

                t0 = perf_counter()
                assignment.execute()
                elapsed_list.append(perf_counter() - t0)

                iters_run_list.append(int(assignment.assignment.iter))
                rgap_list.append(float(assignment.assignment.rgap))
                cores_used = int(assignment.cores)

                total_loads = assigclass.results.total_link_loads
                link_load_sums.append(float(total_loads.sum()))
                link_load_maxs.append(float(total_loads.max()))

            import statistics as _stats

            return {
                "backend": backend,
                "algorithm": algorithm,
                "repetitions": len(elapsed_list),
                "elapsed_seconds_list": elapsed_list,
                "elapsed_seconds_min": float(min(elapsed_list)),
                "elapsed_seconds_median": float(_stats.median(elapsed_list)),
                "elapsed_seconds_mean": float(_stats.mean(elapsed_list)),
                "elapsed_seconds_max": float(max(elapsed_list)),
                "elapsed_seconds_stdev": (
                    float(_stats.stdev(elapsed_list)) if len(elapsed_list) > 1 else 0.0
                ),
                "iterations_requested": iterations,
                "iterations_run_list": iters_run_list,
                "rgap_list": rgap_list,
                "rgap_target": rgap_target,
                "cores": cores_used,
                "num_zones": int(graph.num_zones),
                "num_links": int(graph.num_links),
                "matrix": matrix_name,
                "mode": mode,
                "link_load_sum_last": link_load_sums[-1],
                "link_load_max_last": link_load_maxs[-1],
                "smoke": smoke,
            }
        finally:
            matrix.close()
    finally:
        project.close()


def _spawn_child(
    backend: str,
    args: argparse.Namespace,
    matrix_name: str,
    iterations: int,
    cores: int,
) -> Dict:
    script = Path(__file__).resolve()
    env = os.environ.copy()
    if backend == "smartq":
        env["AEQ_SMARTQ"] = "1"
    else:
        env.pop("AEQ_SMARTQ", None)
    if "AEQ_BUCKET_QUEUE" in env:
        sys.stderr.write(
            "warning: AEQ_BUCKET_QUEUE is set in the parent environment; "
            "it does not affect TrafficAssignment but may indicate a stale "
            "configuration.\n"
        )

    cmd = [
        sys.executable,
        str(script),
        "--single-state",
        "--backend",
        backend,
        "--model-path",
        str(args.model_path),
        "--mode",
        args.mode,
        "--matrix-name",
        matrix_name,
        "--iterations",
        str(iterations),
        "--cores",
        str(cores),
        "--rgap-target",
        str(args.rgap_target),
        "--algorithm",
        args.algorithm,
        "--repetitions",
        str(args.repetitions),
    ]
    if args.smoke:
        cmd.append("--smoke")

    print(
        f"\n=== Running backend={backend} (iterations={iterations}, cores={cores}) ===",
        flush=True,
    )
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"Child process for backend={backend} failed: rc={proc.returncode}")

    last_line = ""
    for line in proc.stdout.splitlines():
        if line.strip():
            last_line = line
    if not last_line:
        raise RuntimeError(f"Child for backend={backend} produced no JSON output. stderr:\n{proc.stderr}")
    return json.loads(last_line)


def _print_summary(rows: List[Dict], algorithm: str) -> None:
    print(f"\n=== {algorithm.upper()} Traffic Assignment: 4-ary heap vs Smart Queue ===")
    headers = [
        "backend",
        "reps",
        "min_s",
        "median_s",
        "mean_s",
        "max_s",
        "stdev_s",
        "iters",
        "rgap",
        "cores",
        "speedup_median",
    ]
    print(" | ".join(f"{h:>14}" for h in headers))
    print("-" * (15 * len(headers) + len(headers) - 1))

    base = next(
        (r["elapsed_seconds_median"] for r in rows if r["backend"] == "4ary"),
        None,
    )
    for row in rows:
        if base and base > 0 and row["backend"] != "4ary":
            speedup = (base - row["elapsed_seconds_median"]) / base * 100.0
            speedup_str = f"{speedup:+.2f}%"
        else:
            speedup_str = "-"
        iters_run = row["iterations_run_list"]
        rgap = row["rgap_list"]
        iters_str = (
            str(iters_run[0])
            if all(v == iters_run[0] for v in iters_run)
            else f"{min(iters_run)}-{max(iters_run)}"
        )
        rgap_str = f"{max(rgap):.3e}" if rgap else "-"
        print(
            " | ".join(
                f"{v:>14}"
                for v in [
                    row["backend"],
                    str(row["repetitions"]),
                    f"{row['elapsed_seconds_min']:.3f}",
                    f"{row['elapsed_seconds_median']:.3f}",
                    f"{row['elapsed_seconds_mean']:.3f}",
                    f"{row['elapsed_seconds_max']:.3f}",
                    f"{row['elapsed_seconds_stdev']:.3f}",
                    iters_str,
                    rgap_str,
                    str(row["cores"]),
                    speedup_str,
                ]
            )
        )


def _print_sweep_summary(rows: List[Dict], algorithm: str) -> None:
    print(f"\n=== {algorithm.upper()} core sweep: 4-ary heap vs Smart Queue ===")
    headers = [
        "cores",
        "4ary_mean_s",
        "4ary_median_s",
        "smartq_mean_s",
        "smartq_median_s",
        "speedup_mean",
        "speedup_median",
    ]
    print(" | ".join(f"{h:>16}" for h in headers))
    print("-" * (17 * len(headers) + len(headers) - 1))

    by_cores: Dict[int, Dict[str, Dict]] = {}
    for r in rows:
        by_cores.setdefault(r["cores"], {})[r["backend"]] = r

    for cores in sorted(by_cores):
        pair = by_cores[cores]
        h = pair.get("4ary")
        s = pair.get("smartq")
        if h is None or s is None:
            continue
        sp_mean = (
            (h["elapsed_seconds_mean"] - s["elapsed_seconds_mean"])
            / h["elapsed_seconds_mean"]
            * 100.0
        )
        sp_med = (
            (h["elapsed_seconds_median"] - s["elapsed_seconds_median"])
            / h["elapsed_seconds_median"]
            * 100.0
        )
        print(
            " | ".join(
                f"{v:>16}"
                for v in [
                    str(cores),
                    f"{h['elapsed_seconds_mean']:.3f}",
                    f"{h['elapsed_seconds_median']:.3f}",
                    f"{s['elapsed_seconds_mean']:.3f}",
                    f"{s['elapsed_seconds_median']:.3f}",
                    f"{sp_mean:+.2f}%",
                    f"{sp_med:+.2f}%",
                ]
            )
        )


def _write_outputs(rows: List[Dict], output_dir: Path, stamp: str, algorithm: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"heap_vs_smartq_{algorithm}_{stamp}"
    json_path = output_dir / f"{base}.json"
    csv_path = output_dir / f"{base}.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        # Flatten list-valued fields for CSV.
        flat_rows = []
        for r in rows:
            flat = dict(r)
            flat["elapsed_seconds_list"] = ";".join(f"{v:.6f}" for v in r["elapsed_seconds_list"])
            flat["iterations_run_list"] = ";".join(str(v) for v in r["iterations_run_list"])
            flat["rgap_list"] = ";".join(f"{v:.6e}" for v in r["rgap_list"])
            flat_rows.append(flat)
        fieldnames = list(flat_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)
    print("\nWrote outputs:")
    print(f"  {json_path}")
    print(f"  {csv_path}")


def _maybe_rebuild() -> None:
    print("Rebuilding aequilibrae extensions (pip install -e .[dev]) ...", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
        cwd=str(_repo_root()),
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("Rebuild failed")


def _correctness_check(rows: List[Dict]) -> None:
    if len(rows) < 2:
        return
    base = next((r for r in rows if r["backend"] == "4ary"), None)
    other = next((r for r in rows if r["backend"] == "smartq"), None)
    if base is None or other is None:
        return
    s_diff = abs(base["link_load_sum_last"] - other["link_load_sum_last"])
    s_rel = s_diff / max(abs(base["link_load_sum_last"]), 1.0)
    m_diff = abs(base["link_load_max_last"] - other["link_load_max_last"])
    m_rel = m_diff / max(abs(base["link_load_max_last"]), 1.0)
    print("\nCorrectness (link load aggregates, last repetition):")
    print(
        f"  sum:  4ary={base['link_load_sum_last']:.6e}  "
        f"smartq={other['link_load_sum_last']:.6e}  rel_diff={s_rel:.3e}"
    )
    print(
        f"  max:  4ary={base['link_load_max_last']:.6e}  "
        f"smartq={other['link_load_max_last']:.6e}  rel_diff={m_rel:.3e}"
    )
    tol = 1e-6
    if s_rel > tol or m_rel > tol:
        print(f"  WARNING: relative diff above {tol:.0e}; small tie-breaking deviations are expected")


def _orchestrate(args: argparse.Namespace) -> None:
    if args.rebuild:
        _maybe_rebuild()

    matrix_name = _resolve_matrix_name(args.model_path, args.mode, args.matrix_name or None)
    iterations = 2 if args.smoke else args.iterations

    if args.cores_list.strip():
        cores_sweep = [int(v) for v in args.cores_list.split(",") if v.strip()]
    else:
        cores_sweep = [args.cores]

    print("Configuration:")
    print(f"  model_path     : {args.model_path}")
    print(f"  mode           : {args.mode}")
    print(f"  matrix         : {matrix_name}")
    print(f"  algorithm      : {args.algorithm}")
    print(f"  iterations     : {iterations}{' (smoke)' if args.smoke else ''}")
    print(f"  repetitions    : {args.repetitions}")
    print(f"  rgap_target    : {args.rgap_target}")
    print(
        "  cores sweep    : "
        + ", ".join(str(c) if c > 0 else "all" for c in cores_sweep)
    )

    backends = ["4ary", "smartq"]
    all_rows: List[Dict] = []
    t_total = time.perf_counter()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.smoke:
        stamp = f"smoke_{stamp}"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / f"heap_vs_smartq_{args.algorithm}_{stamp}.partial.json"

    for cores in cores_sweep:
        print(f"\n##### cores = {cores if cores > 0 else 'all'} #####", flush=True)
        rows: List[Dict] = []
        for backend in backends:
            result = _spawn_child(backend, args, matrix_name, iterations, cores)
            rows.append(result)
            all_rows.append(result)
            # Persist progress after every backend cell so a killed run
            # does not lose hours of work.
            partial_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
            print(
                f"  -> backend={backend} reps={result['repetitions']} "
                f"min={result['elapsed_seconds_min']:.3f}s "
                f"median={result['elapsed_seconds_median']:.3f}s "
                f"mean={result['elapsed_seconds_mean']:.3f}s "
                f"max={result['elapsed_seconds_max']:.3f}s "
                f"stdev={result['elapsed_seconds_stdev']:.3f}s",
                flush=True,
            )
        _print_summary(rows, args.algorithm)
        _correctness_check(rows)

    print(f"\nTotal wall time: {time.perf_counter() - t_total:.1f}s")

    if len(cores_sweep) > 1:
        _print_sweep_summary(all_rows, args.algorithm)

    _write_outputs(all_rows, output_dir, stamp, args.algorithm)
    if partial_path.exists():
        try:
            partial_path.unlink()
        except OSError:
            pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(r"D:\tmp\Chicago_aeq"),
        help="Path to AequilibraE project folder",
    )
    parser.add_argument("--mode", default="c", help="Graph mode")
    parser.add_argument("--iterations", type=int, default=100, help="max_iter for the assignment")
    parser.add_argument(
        "--matrix-name",
        default="",
        help="Matrix to use; auto-detected if empty (prefers 'demand')",
    )
    parser.add_argument("--cores", type=int, default=0, help="Cores (0 = all)")
    parser.add_argument(
        "--cores-list",
        default="",
        help="Comma-separated list of core counts to sweep (e.g. '2,4,8,16,32'). "
        "When set, --cores is ignored and the benchmark runs once per value.",
    )
    parser.add_argument(
        "--algorithm",
        default="msa",
        choices=["msa", "frank-wolfe", "fw", "cfw", "bfw"],
        help="Traffic-assignment algorithm to benchmark",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of timed repetitions per backend (per core count)",
    )
    parser.add_argument(
        "--rgap-target",
        type=float,
        default=1e-12,
        help="rgap target; set very low so MSA does not early-exit before the requested iteration count",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarking") / "results",
        help="Directory for JSON/CSV outputs",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Run pip install -e .[dev] before benchmarking",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run with iterations=2 and report aggregate-load deltas as a correctness check",
    )

    parser.add_argument("--single-state", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.single_state:
        if not args.backend:
            raise RuntimeError("--single-state requires --backend")
        result = _run_single_backend(
            backend=args.backend,
            model_path=args.model_path,
            mode=args.mode,
            matrix_name=args.matrix_name,
            iterations=args.iterations,
            cores=args.cores,
            rgap_target=args.rgap_target,
            smoke=args.smoke,
            algorithm=args.algorithm,
            repetitions=args.repetitions,
        )
        print(json.dumps(result))
        return

    _orchestrate(args)


if __name__ == "__main__":
    # statistics is imported above for future use (median/mean across reps).
    _ = statistics
    main()
