#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heap benchmark binary repeatedly and plot timings")
    parser.add_argument("--binary", default="./a.out", help="Path to benchmark binary")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[10000, 50000, 100000, 250000, 500000, 900000],
        help="Element counts to benchmark",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Runs per size")
    parser.add_argument("--output", default="heap_benchmark_plot.png", help="Output plot image path")
    parser.add_argument("--save-csv", default="heap_benchmark_results.csv", help="Output CSV path")
    return parser.parse_args()


def run_once(binary: str, size: int) -> dict:
    proc = subprocess.run(
        [binary, str(size), "--csv"],
        check=True,
        text=True,
        capture_output=True,
    )

    line = proc.stdout.strip().splitlines()[-1]
    parts = line.split(",")
    if len(parts) == 10:
        return {
            "elements": int(parts[0]),
            "decrease_key_ops": int(parts[1]),
            "std_ms": float(parts[2]),
            "adapter_ms": float(parts[3]),
            "fourary_ms": float(parts[4]),
            "pairing_ms": float(parts[5]),
            "adapter_vs_std": float(parts[6]),
            "fourary_vs_std": float(parts[7]),
            "pairing_vs_std": float(parts[8]),
            "status": parts[9],
        }
    raise RuntimeError(f"Unexpected CSV row: {line}")


def average_rows(rows: list[dict]) -> dict:
    n = len(rows)
    first = rows[0]
    return {
        "elements": first["elements"],
        "decrease_key_ops": first["decrease_key_ops"],
        "std_ms": sum(r["std_ms"] for r in rows) / n,
        "adapter_ms": sum(r["adapter_ms"] for r in rows) / n,
        "fourary_ms": sum(r["fourary_ms"] for r in rows) / n,
        "pairing_ms": sum(r["pairing_ms"] for r in rows) / n,
        "adapter_vs_std": sum(r["adapter_vs_std"] for r in rows) / n,
        "fourary_vs_std": sum(r["fourary_vs_std"] for r in rows) / n,
        "pairing_vs_std": sum(r["pairing_vs_std"] for r in rows) / n,
        "status": "ok",
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    fieldnames = [
        "elements",
        "decrease_key_ops",
        "std_ms",
        "adapter_ms",
        "fourary_ms",
        "pairing_ms",
        "adapter_vs_std",
        "fourary_vs_std",
        "pairing_vs_std",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(path: Path, rows: list[dict]) -> None:
    x = [r["elements"] for r in rows]
    y_std = [r["std_ms"] for r in rows]
    y_adapter = [r["adapter_ms"] for r in rows]
    y_four = [r["fourary_ms"] for r in rows]
    y_pair = [r["pairing_ms"] for r in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_std, marker="o", label="std::priority_queue")
    plt.plot(x, y_adapter, marker="o", label="StdPriorityQueueAdapter")
    plt.plot(x, y_four, marker="o", label="FourAryHeap")
    plt.plot(x, y_pair, marker="o", label="PairingHeap")
    plt.xlabel("Number of elements")
    plt.ylabel("Time (ms)")
    plt.title("Heap benchmark (average over repeats)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)


def main() -> int:
    args = parse_args()

    binary = args.binary
    repeats = args.repeats
    sizes = args.sizes

    if repeats <= 0:
        print("--repeats must be > 0", file=sys.stderr)
        return 2
    if any(size <= 0 for size in sizes):
        print("All --sizes must be > 0", file=sys.stderr)
        return 2

    raw_rows: list[dict] = []
    averaged_rows: list[dict] = []

    for size in sizes:
        size_rows = []
        print(f"Running size={size} repeats={repeats}")
        for rep in range(repeats):
            row = run_once(binary, size)
            if row["status"] != "ok":
                raise RuntimeError(f"Benchmark reported non-ok status: {row}")
            size_rows.append(row)
            raw_rows.append(row)
            print(
                f"  rep={rep + 1}: std={row['std_ms']:.3f}ms adapter={row['adapter_ms']:.3f}ms "
                f"four={row['fourary_ms']:.3f}ms pair={row['pairing_ms']:.3f}ms"
            )

        averaged = average_rows(size_rows)
        averaged_rows.append(averaged)
        print(
            f"  avg: std={averaged['std_ms']:.3f}ms adapter={averaged['adapter_ms']:.3f}ms "
            f"four={averaged['fourary_ms']:.3f}ms pair={averaged['pairing_ms']:.3f}ms"
        )

    save_csv(Path(args.save_csv), averaged_rows)
    plot_rows(Path(args.output), averaged_rows)

    print(f"Saved CSV: {args.save_csv}")
    print(f"Saved plot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
