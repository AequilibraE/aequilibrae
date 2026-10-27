"""Time transit graph building and Spiess & Florian assignment.

Two cases, neither needing an external model:

- ``coquimbo``: the whole pipeline on the Coquimbo example (graph build, save and reload, then assignment of one trip
  between every pair of zones with an increasing number of skims).
- ``grid``: ``HyperpathGenerating.assign`` on Bell's random grid, large enough that kernel time dominates. Every
  assignment waits for the progress bridge, which polls every 500 ms, so small cases only measure that floor.

Usage::

    python benchmarking/transit_performance.py --case coquimbo grid --threads 1 0 --output transit_performance.csv

``--threads 0`` uses every available thread.
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import HyperpathGenerating, TransitAssignment, TransitClass
from aequilibrae.utils.create_example import create_example

SKIMS = ["trav_time", "boardings", "in_vehicle_trav_time", "waiting_time", "transfers", "access_trav_time"]


def best_of(repeats, function):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times)


def coquimbo(threads: int, repeats: int) -> list[dict]:
    rows = []
    project = create_example(str(Path(tempfile.mkdtemp()) / "coquimbo"), "coquimbo")
    transit = project.scenario.transit

    start = time.perf_counter()
    builder = transit.create_graph(with_walking_edges=False, connector_method="overlapping_regions")
    rows.append({"case": "coquimbo", "step": "build graph", "seconds": time.perf_counter() - start})

    start = time.perf_counter()
    builder.create_line_geometry(method="direct")
    transit.save_graphs()
    transit.load()
    rows.append({"case": "coquimbo", "step": "geometry, save and load", "seconds": time.perf_counter() - start})

    graph = transit.graphs[1].to_transit_graph()
    zones = len(graph.centroids)
    matrix = AequilibraeMatrix(index=graph.centroids)
    matrix["pt"] = np.ones((zones, zones)) - np.eye(zones)
    transit_class = TransitClass(name="pt", graph=graph, matrix=matrix)

    for count in range(len(SKIMS) + 1):
        assignment = TransitAssignment()
        assignment.add_class(transit_class)
        assignment.set_time_field("trav_time")
        assignment.set_frequency_field("freq")
        assignment.set_skimming_fields(SKIMS[:count])
        assignment.set_algorithm("os")
        assignment.set_cores(threads)

        seconds = best_of(repeats, assignment.execute)
        if count:
            assert len(assignment.get_skim_results()["pt"]) == count
        rows.append({"case": "coquimbo", "step": f"assign, {count} skims", "threads": threads, "seconds": seconds})

    project.close()
    return rows


def bell_grid(n: int, zones: int, seed: int = 124):
    """Bell's grid with random costs and frequencies, and one trip between every pair of random zones."""
    i, j = (values.ravel() for values in np.meshgrid(np.arange(n - 1), np.arange(n), indexing="ij"))
    # Rightward edges along each row, then upward edges along each column
    tail = np.concatenate((i + j * n, j + i * n))
    head = np.concatenate((i + 1 + j * n, j + (i + 1) * n))
    rng = np.random.default_rng(seed)
    edges = pd.DataFrame(
        {
            "tail": tail.astype(np.uint32),
            "head": head.astype(np.uint32),
            "trav_time": rng.uniform(0.0, 1.0, len(tail)),
            "freq": 0.1 / rng.uniform(0.01, 1.0, len(tail)),
            "length": rng.uniform(0.0, 5.0, len(tail)),
        }
    )
    centroids = np.sort(rng.choice(n * n, size=zones, replace=False))
    o, d = np.meshgrid(centroids, centroids, indexing="ij")
    keep = (o != d).ravel()
    demand = (o.ravel()[keep].astype(np.uint32), d.ravel()[keep].astype(np.uint32), np.ones(keep.sum()))
    return edges, centroids, demand


def grid(threads: int, repeats: int, n: int = 300, zones: int = 1000) -> list[dict]:
    rows = []
    edges, centroids, demand = bell_grid(n, zones)
    for skims in (None, ["trav_time", "length"]):
        hyperpath = HyperpathGenerating(
            edges,
            skim_cols=skims,
            o_vert_ids=centroids,
            d_vert_ids=centroids,
            nodes_to_indices=np.arange(n * n, dtype=np.int64),
        )
        seconds = best_of(repeats, lambda h=hyperpath: h.assign(*demand, threads=threads))
        step = f"assign {len(edges)} edges to {zones} destinations, {len(skims or [])} skims"
        rows.append({"case": "grid", "step": step, "threads": threads, "seconds": seconds})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", nargs="+", choices=["coquimbo", "grid"], default=["coquimbo", "grid"])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("transit_performance.csv"))
    args = parser.parse_args()

    cases = {"coquimbo": coquimbo, "grid": grid}
    rows = []
    for name in args.case:
        for threads in args.threads:
            for row in cases[name](threads, args.repeats):
                row.setdefault("threads", threads)
                print(row, flush=True)
                rows.append(row)

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
