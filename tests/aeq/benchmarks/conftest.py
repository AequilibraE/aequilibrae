"""Shared test infrastructure for TNTP validation tests."""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph, TrafficAssignment
from aequilibrae.paths.traffic_class import TrafficClass
from scipy.stats import linregress

import pytest as pytest


# pytest .\tests\aeq\benchmarks\* -- benchmark

# python plot_benchmarks.py [--convergence] [--compare-flow] [--reports-dir PATH] [--x-axis {time,iterations}]

# $env:TNTP_ROOT="C:\Users\jake\src\aequilibrae\TransportationNetworks"

"""
The path to the repo "TransportationNetworks" must be set using the environment variable TNTP_ROOT.
Found at https://github.com/bstabler/TransportationNetworks

The output directory where the csv results files will be placed also needs to be set
using the environment variable BENCHMARK_REPORTS_DIR

e.g.
export TNTP_ROOT="../../TransportationNetworks"
export BENCHMARK_REPORTS_DIR="./tests/aeq/benchmarks/_convergence_reports"

To run these tests, the arugment --benchmark needs to be specified:
pytest ./tests/aeq/benchmarks/* --benchmark

So these tests are skipped with:
pytest .

"""


METHODS = ["msa", "frank-wolfe", "cfw", "bfw"]
ITERATIONS = 1000
RGAP_TARGET = 1e-15

R2_MINIMUM = 0.95
INTERCEPT_MINIMUM = 1e3


@pytest.fixture(scope="module")
def tntp_root():
    return Path(os.environ["TNTP_ROOT"])


@pytest.fixture(scope="module")
def tntp_matrix(model_folder, model_stub):
    mat = load_tntp_matrix(model_folder, model_stub)
    yield mat
    mat.close()


@pytest.fixture(scope="module")
def tntp_graph(model_folder, model_stub, tntp_matrix):
    return load_tntp_graph(model_folder, model_stub, tntp_matrix.index)


@pytest.fixture(scope="module")
def tntp_reference(model_folder, model_stub):
    return load_known_results(model_folder, model_stub)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "benchmark: marks tests as a benchmark (select with '\"--benchmark\"')",
    )


def pytest_addoption(parser):
    parser.addoption("--benchmark", action="store_true", default=False, help="run benchmarking")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--benchmark"):
        # --benchmark given in cli: do not skip slow benchmark tests
        return
    skip_benchmark = pytest.mark.skip(reason="need --benchmark option to run")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)


def parse_tntp_header(folder: Path, model_stub: str) -> dict:
    result = {}
    with open(folder / f"{model_stub}_net.tntp") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("<") and ">" in line:
                key, _, val = line[1:].partition(">")
                try:
                    result[key.strip()] = int(val.strip())
                except ValueError:
                    pass
            elif line.startswith("~"):
                break
    return result


def load_tntp_matrix(folder: Path, model_stub: str) -> AequilibraeMatrix:
    omx_name = folder / f"{model_stub}_trips.omx"
    if omx_name.exists():
        mat = AequilibraeMatrix()
        mat.load(omx_name)
        mat.computational_view()
        return mat

    matfile = str(folder / f"{model_stub}_trips.tntp")
    with open(matfile, "r") as fh:
        all_rows = fh.read()
    blocks = all_rows.split("Origin")[1:]
    matrix = {}
    for k in range(len(blocks)):
        orig = blocks[k].split("\n")
        dests = orig[1:]
        orig = int(orig[0])
        d = [eval("{" + a.replace(";", ",").replace(" ", "") + "}") for a in dests]
        destinations = {}
        for i in d:
            destinations = {**destinations, **i}
        matrix[orig] = destinations
    zones = max(matrix.keys())
    index = np.arange(zones) + 1
    mat_data = np.zeros((zones, zones))
    for i in range(zones):
        for j in range(zones):
            mat_data[i, j] = matrix[i + 1].get(j + 1, 0)

    mat = AequilibraeMatrix()
    mat.create_empty(zones=zones, matrix_names=["matrix"], memory_only=True)
    mat.matrix["matrix"][:, :] = mat_data[:, :]
    mat.index[:] = index[:]
    mat.computational_view(["matrix"])
    mat.export(str(omx_name))
    return mat


def load_tntp_graph(folder: Path, model_stub: str, centroids: np.ndarray) -> Graph:
    header = parse_tntp_header(folder, model_stub)
    first_thru_node = header.get("FIRST THRU NODE", 2)

    net = pd.read_csv(folder / f"{model_stub}_net.tntp", skiprows=7, sep="\t")
    cols = [
        "init_node",
        "term_node",
        "free_flow_time",
        "capacity",
        "b",
        "power",
        "length",
    ]
    if "toll" in net.columns:
        cols.append("toll")
    network = net[cols].copy()
    new_cols = [
        "a_node",
        "b_node",
        "free_flow_time",
        "capacity",
        "b",
        "power",
        "length",
    ]
    if "toll" in net.columns:
        new_cols.append("toll")
    network.columns = new_cols
    network = network.assign(direction=1)
    network["link_id"] = network.index + 1
    network["free_flow_time"] = network["free_flow_time"].astype(np.float64)

    g = Graph()
    g.cost = net["free_flow_time"].values
    g.capacity = net["capacity"].values
    g.free_flow_time = net["free_flow_time"].values

    g.network = network
    g.network.loc[g.network["power"] < 1, "power"] = 1
    g.network.loc[g.network["free_flow_time"] == 0, "free_flow_time"] = 0.01
    g.prepare_graph(centroids)
    g.set_graph("free_flow_time")
    g.set_skimming(["free_flow_time"])
    g.set_blocked_centroid_flows(first_thru_node > 1)
    return g


def load_known_results(folder: Path, model_stub: str) -> pd.DataFrame:
    path = folder / f"{model_stub}_flow.tntp"
    with open(path) as fh:
        first_line = fh.readline().strip()
    skiprows = 8 if first_line.startswith("<") else 0
    df = pd.read_csv(path, skiprows=skiprows, sep=r"\s+", engine="python")
    df = df.loc[:, ~df.columns.str.strip().isin([";", ""])]
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("tail", "from"):
            col_map[c] = "a_node"
        elif cl in ("head", "to"):
            col_map[c] = "b_node"
        elif cl in ("volume",):
            col_map[c] = "TNTP Solution"
        elif cl in ("cost",):
            col_map[c] = "cost"
    df = df.rename(columns=col_map)
    return df[["a_node", "b_node", "TNTP Solution"]].dropna()


def assert_flow_regression(
    aeq_flows: np.ndarray,
    tntp_flows: np.ndarray,
    *,
    r2_limit: float = R2_MINIMUM,
    int_limit: float = INTERCEPT_MINIMUM,
):
    """Assert R² >= r2_limit and |intercept| <= int_limit."""
    aeq_flows = np.asarray(aeq_flows, dtype=np.float64)
    tntp_flows = np.asarray(tntp_flows, dtype=np.float64)
    reg = linregress(tntp_flows, aeq_flows)
    r2 = reg.rvalue**2
    assert r2 >= r2_limit, f"R²={r2:.6f} below threshold {r2_limit}"
    assert abs(reg.intercept) <= int_limit, f"intercept={reg.intercept:.4f} exceeds {int_limit}"


def save_convergence_report(trials: list[pd.DataFrame], model_name: str, algorithm: str):
    """Save multi-trial convergence reports as a single CSV."""
    benchmark_reports_dir = Path(os.environ["BENCHMARK_REPORTS_DIR"])
    benchmark_reports_dir.mkdir(exist_ok=True)
    parts = []
    for i, report in enumerate(trials):
        out = report.copy()
        out["trial"] = i
        parts.append(out)
    combined = pd.concat(parts, ignore_index=True)
    combined["model"] = model_name
    combined["algorithm"] = algorithm
    combined.to_csv(benchmark_reports_dir / f"{model_name}_{algorithm}.csv", index=False)


def save_flow_results_with_nodes(results_with_nodes: pd.DataFrame, model_name: str, algorithm: str):
    """Save flow results."""
    benchmark_reports_dir = Path(os.environ["BENCHMARK_REPORTS_DIR"])
    benchmark_reports_dir.mkdir(exist_ok=True)

    results_with_nodes.to_parquet(benchmark_reports_dir / f"{model_name}_{algorithm}_results_with_nodes.parquet")


def run_validation(
    benchmark,
    graph,
    matrix,
    reference,
    stub,
    algorithm,
    *,
    known_pass=None,
):
    """Configure, run, bench, validate, and plot an assignment against TNTP reference.

    known_pass: optional dict mapping algorithm -> {"r2": float, "intercept": float}
    with the best thresholds known to pass for that algorithm at full convergence.
    If the globally set R2_MINIMUM / INTERCEPT_MINIMUM are weaker, a warning is emitted
    and the looser (known) threshold is used for the assertion.
    """
    r2_limit = R2_MINIMUM
    int_limit = INTERCEPT_MINIMUM

    if known_pass and algorithm in known_pass:
        kp = known_pass[algorithm]
        kp_r2 = kp["r2"]
        kp_int = kp["intercept"]

        if R2_MINIMUM < kp_r2:
            warnings.warn(
                f"{stub}/{algorithm}: R² threshold {R2_MINIMUM} is weaker than "
                f"known passing {kp_r2}; using {R2_MINIMUM}",
                stacklevel=2,
            )
        else:
            r2_limit = min(R2_MINIMUM, kp_r2)

        if INTERCEPT_MINIMUM > kp_int:
            warnings.warn(
                f"{stub}/{algorithm}: intercept threshold {INTERCEPT_MINIMUM} is looser than "
                f"known passing {kp_int}; using {INTERCEPT_MINIMUM}",
                stacklevel=2,
            )
        else:
            int_limit = max(INTERCEPT_MINIMUM, kp_int)

    # --- factory for a clean TrafficAssignment per trial ---
    def _make_assignment():
        tc = TrafficClass("car", graph, matrix)
        if "toll" in graph.network.columns:
            tc.set_fixed_cost("toll")
            tc.set_vot(1.0)

        a = TrafficAssignment()
        a.set_classes([tc])
        a.set_vdf("BPR")
        a.set_vdf_parameters({"alpha": "b", "beta": "power"})
        a.set_capacity_field("capacity")
        a.set_time_field("free_flow_time")
        a.max_iter = ITERATIONS
        a.rgap_target = RGAP_TARGET
        a.set_algorithm(algorithm)
        return a

    trials: list[pd.DataFrame] = []

    def _benchmarked():
        assig: TrafficAssignment = _make_assignment()
        assig.execute()
        trials.append(assig.report())
        return assig

    assig: TrafficAssignment = benchmark(_benchmarked)
    assert isinstance(assig, TrafficAssignment)

    link_lookup = graph.network[["link_id", "a_node", "b_node"]].set_index("link_id")
    results_with_nodes = (
        assig.results()[["PCE_AB"]].join(link_lookup).merge(reference, on=["a_node", "b_node"], how="right")
    )
    assert len(results_with_nodes) > 0, "No matching links between assignment results and TNTP reference"

    save_flow_results_with_nodes(results_with_nodes, stub, algorithm)

    save_convergence_report(trials, stub, algorithm)

    assert_flow_regression(
        results_with_nodes["PCE_AB"].values,
        results_with_nodes["TNTP Solution"].values,
        r2_limit=r2_limit,
        int_limit=int_limit,
    )
