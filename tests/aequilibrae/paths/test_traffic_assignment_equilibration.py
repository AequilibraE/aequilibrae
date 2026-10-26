import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aequilibrae import TrafficAssignment, TrafficClass
from aequilibrae.paths.vdf import bpr
from aequilibrae.utils.logging_utils import basic_config


@pytest.fixture
def configuration(sioux_falls_test):
    project = sioux_falls_test
    demand = project.matrices.get_matrix("omx2", subset=["matrix"])
    return {
        "network": project.network,
        "classes": [
            TrafficClass(
                name="car",
                mode="c",
                cost="free_flow_time",
                demand=demand,
                skimming=["free_flow_time", "distance"],
                block_centroid_flows=False,
            )
        ],
        "time_field": "free_flow_time",
        "capacity_field": "capacity",
        "vdf": bpr,
        "name_mapping": {"alpha": "b", "beta": "power"},
        "algorithm": "bfw",
        "project": project,
    }


@pytest.fixture
def file_logging():
    logger = logging.getLogger("aequilibrae")
    handler = basic_config(level=logging.INFO)
    assert handler is not None
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


@pytest.mark.parametrize("pce", [1.0, 2.5])
def test_max_iterations_returns_the_iterate_with_reported_gap(configuration, pce):
    configuration["classes"] = [replace(configuration["classes"][0], pce=pce)]
    assignment = TrafficAssignment(**configuration)
    assignment.execute(max_iterations=2, cores=1)
    state = assignment._class_states[0]
    algorithm = assignment.assignment
    report = algorithm.convergence_report
    # Both result groups remain in demand units. The optimizer applies PCE to both.
    unit_cost = algorithm.congested_time + state.fixed_cost
    expected_current_cost = np.sum(unit_cost * state.results.total_link_loads * pce)
    expected_aon_cost = np.sum(unit_cost * state._aon_results.total_link_loads * pce)
    expected_rgap = abs(expected_current_cost - expected_aon_cost) / expected_current_cost

    assert np.isclose(algorithm.rgap, expected_rgap)
    assert np.isclose(report["rgap"][-1], expected_rgap)
    assert np.isnan(report["alpha"][-1])
    assert all(np.isnan(report[key][-1]) for key in ("beta0", "beta1", "beta2"))
    assert len({len(values) for values in report.values()}) == 1


def test_execute_and_save_results(configuration, file_logging):
    project = configuration["project"]
    with project.db_connection as conn:
        results = pd.read_sql("select volume from links order by link_id", conn)

    assignment = TrafficAssignment(**(configuration | {"algorithm": "msa"}))
    assignment.execute(max_iterations=10, cores=1)

    msa10_rgap = assignment.assignment.rgap
    correl = np.corrcoef(assignment.class_results["car"].total_link_loads, results.volume.values)[0, 1]
    assert 0.8 < correl

    assignment.execute(max_iterations=500, rgap=0.001, cores=1)
    msa25_rgap = assignment.assignment.rgap
    correl = np.corrcoef(assignment.class_results["car"].total_link_loads, results.volume)[0, 1]
    assert 0.98 < correl

    assignment = TrafficAssignment(**(configuration | {"algorithm": "frank-wolfe"}))
    assignment.execute(max_iterations=500, rgap=0.001, cores=1)
    fw25_rgap = assignment.assignment.rgap
    fw25_iters = assignment.assignment.iter
    correl = np.corrcoef(assignment.class_results["car"].total_link_loads, results.volume)[0, 1]
    assert 0.99 < correl

    assignment = TrafficAssignment(**(configuration | {"algorithm": "cfw"}))
    assignment.execute(max_iterations=500, rgap=0.001, cores=1)
    cfw25_rgap = assignment.assignment.rgap
    cfw25_iters = assignment.assignment.iter
    correl = np.corrcoef(assignment.class_results["car"].total_link_loads, results.volume)[0, 1]
    assert 0.995 < correl

    assignment = TrafficAssignment(**configuration)
    assignment.execute(max_iterations=500, rgap=0.001, cores=1)
    bfw25_rgap = assignment.assignment.rgap
    correl = np.corrcoef(assignment.class_results["car"].total_link_loads, results.volume)[0, 1]
    assert 0.999 < correl

    assert msa25_rgap < msa10_rgap
    assert fw25_rgap < msa25_rgap
    assert cfw25_rgap < assignment.rgap_target
    assert bfw25_rgap < assignment.rgap_target
    assert cfw25_iters < fw25_iters

    assignment.save_results("save_to_database")
    assignment.save_skims(matrix_name="all_skims", which_ones="all")

    with pytest.raises(ValueError):
        assignment.save_results("save_to_database")

    log = (Path(project.path_to_file).parent / "aequilibrae.log").read_text()
    assert log.count("Traffic Class specification") > 1
    for algorithm in ("msa", "frank-wolfe", "cfw", "bfw"):
        assert f"'Algorithm': '{algorithm}'" in log
    assert "'matrix_totals': {'matrix': 360600.0}" in log


def test_execute_no_project(configuration):
    project = configuration.pop("project")
    with project.db_connection as conn:
        reference = pd.read_sql("select volume from links order by link_id", conn).volume
    configuration["network"] = project.network.build()
    project.close()
    assignment = TrafficAssignment(**(configuration | {"algorithm": "msa"}))
    assignment.execute(max_iterations=10, cores=1)
    assert assignment.project is None
    assert np.corrcoef(assignment.class_results["car"].total_link_loads, reference)[0, 1] > 0.8
    with pytest.raises(FileNotFoundError):
        assignment.save_results("anything")


@pytest.mark.parametrize("algorithm", ["cfw", "bfw"])
def test_line_search_is_exact_and_uncapped(configuration, algorithm):
    """Test that CFW and BFW take exact line-search steps in [0, 1] that are free to exceed the old 1/sqrt(iter) cap."""
    assignment = TrafficAssignment(**(configuration | {"algorithm": algorithm}))
    assignment.execute(max_iterations=30, rgap=1e-8, cores=1)

    alphas = np.array(assignment.assignment.convergence_report["alpha"], dtype=float)

    assert np.all(np.isfinite(alphas[:-1]))
    assert np.all(alphas[:-1] >= 0.0) and np.all(alphas[:-1] <= 1.0)
    if algorithm == "bfw":
        cap = np.array([1.0 / np.sqrt(i) for i in assignment.assignment.convergence_report["iteration"]])
        assert np.any(alphas[:-1] > cap[:-1]), "the exact line search must be able to exceed the old BFW cap"
    assert np.isfinite(assignment.assignment.rgap)


def test_bfw_conjugacy_defaults_to_approximate(configuration):
    assignment = TrafficAssignment(**configuration)
    assignment.execute(max_iterations=2, cores=1)
    assert assignment.bfw_conjugacy == "approximate"
    assert assignment.assignment.bfw_conjugacy == "approximate"
    assert assignment._config["BFW conjugacy"] == "approximate"


@pytest.mark.parametrize("bad", ["biconjugate", "", 2, None])
def test_bfw_conjugacy_rejects_unknown_methods(configuration, bad):
    assignment = TrafficAssignment(**configuration)
    with pytest.raises(ValueError, match="bfw_conjugacy must be one of"):
        assignment.execute(bfw_conjugacy=bad)


@pytest.mark.parametrize("conjugacy", ["approximate", "exact", "EXACT"])
def test_bfw_reports_conjugacy_diagnostics(configuration, conjugacy):
    assignment = TrafficAssignment(**configuration)
    assignment.execute(max_iterations=25, rgap=1e-8, cores=1, bfw_conjugacy=conjugacy)
    assert assignment.assignment.bfw_conjugacy == conjugacy.lower()
    report = assignment.assignment.convergence_report
    columns = ("conjugacy_prev", "conjugacy_prev2", "hessian_drift", "bfw_clamped")
    assert all(column in report for column in columns)
    assert len({len(values) for values in report.values()}) == 1

    drift = np.array([v for v in report["hessian_drift"] if v is not None], dtype=float)
    measured = drift[np.isfinite(drift)]
    assert measured.size > 0, "at least one iteration should have taken a BFW step"
    # The dropped term is a cosine, so it is bounded even though it need not be small.
    assert np.all(np.abs(measured) <= 1.0 + 1e-9)

    residual = np.array([v for v in report["conjugacy_prev2"] if v is not None], dtype=float)
    residual = np.abs(residual[np.isfinite(residual)])
    assert residual.size > 0
    if conjugacy.lower() == "exact":
        # Only iterations where the clamp did not fire are required to be biconjugate.
        clamped = np.array([v for v in report["bfw_clamped"] if v is not None], dtype=float)
        interior = np.isfinite(clamped) & (clamped == 0.0)
        if interior.any():
            all_residual = np.array(report["conjugacy_prev2"], dtype=float)
            assert np.all(np.abs(all_residual[interior]) < 1e-8)


def test_cfw_does_not_get_conjugacy_columns(configuration):
    assignment = TrafficAssignment(**(configuration | {"algorithm": "cfw"}))
    assignment.execute(max_iterations=3, cores=1)
    assert "conjugacy_prev" not in assignment.assignment.convergence_report
