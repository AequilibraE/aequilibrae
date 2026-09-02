from os.path import isfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aequilibrae import TrafficAssignment, TrafficClass


@pytest.fixture(scope="function")
def project(sioux_falls_test):
    sioux_falls_test.network.build_graphs()
    return sioux_falls_test


@pytest.fixture(scope="function")
def car_graph(project):
    graph = project.network.graphs["c"]
    graph.set_blocked_centroid_flows(False)
    graph.set_graph("free_flow_time")
    return graph


@pytest.fixture(scope="function")
def matrix(project):
    mat = project.matrices.get_matrix("omx2")
    mat.computational_view()
    return mat


@pytest.fixture(scope="function")
def assigclass(car_graph, matrix):
    return TrafficClass("car", car_graph, matrix)


@pytest.fixture(scope="function")
def assignment(project):
    return TrafficAssignment(project)


@pytest.mark.parametrize("pce", [1.0, 2.5])
def test_max_iterations_returns_the_iterate_with_reported_gap(assignment, assigclass, pce):
    assigclass.set_pce(pce)
    assignment.add_class(assigclass)
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("free_flow_time")
    assignment.max_iter = 2
    assignment.set_algorithm("bfw")

    assignment.execute()

    algorithm = assignment.assignment
    report = algorithm.convergence_report
    # The gap is evaluated inside the loop, where class flows are still in PCE units. ``execute`` divides the
    # class results by PCE on the way out, so scale them back to reconstruct the quantity that was reported.
    # The AON results are never rescaled.
    unit_cost = algorithm.congested_time + assigclass.fixed_cost
    expected_current_cost = np.sum(unit_cost * assigclass.results.total_link_loads * pce)
    expected_aon_cost = np.sum(unit_cost * assigclass._aon_results.total_link_loads)
    expected_rgap = abs(expected_current_cost - expected_aon_cost) / expected_current_cost

    assert np.isclose(algorithm.rgap, expected_rgap)
    assert np.isclose(report["rgap"][-1], expected_rgap)
    assert np.isnan(report["alpha"][-1])
    assert all(np.isnan(report[key][-1]) for key in ("beta0", "beta1", "beta2"))
    assert len({len(values) for values in report.values()}) == 1


@pytest.mark.parametrize("matrix_type", ["memmap", "memonly"])
def test_execute_and_save_results(project, assignment, assigclass, car_graph, matrix, matrix_type):
    if matrix_type == "memonly":
        matrix = matrix.copy(memory_only=True)

    with project.db_connection as conn:
        results = pd.read_sql("select volume from links order by link_id", conn)

    proj = assignment.project
    assignment.add_class(assigclass)
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("free_flow_time")
    assignment.max_iter = 10
    assignment.set_algorithm("msa")
    assignment.execute()

    msa10_rgap = assignment.assignment.rgap
    correl = np.corrcoef(assigclass.results.total_link_loads, results.volume.values)[0, 1]
    assert 0.8 < correl

    assignment.max_iter = 500
    assignment.rgap_target = 0.001
    assignment.set_algorithm("msa")
    assignment.execute()
    msa25_rgap = assignment.assignment.rgap
    correl = np.corrcoef(assigclass.results.total_link_loads, results.volume)[0, 1]
    assert 0.98 < correl

    assignment.set_algorithm("frank-wolfe")
    assignment.execute()
    fw25_rgap = assignment.assignment.rgap
    fw25_iters = assignment.assignment.iter
    correl = np.corrcoef(assigclass.results.total_link_loads, results.volume)[0, 1]
    assert 0.99 < correl

    assignment.set_algorithm("cfw")
    assignment.execute()
    cfw25_rgap = assignment.assignment.rgap
    cfw25_iters = assignment.assignment.iter
    correl = np.corrcoef(assigclass.results.total_link_loads, results.volume)[0, 1]
    assert 0.995 < correl

    car_graph.set_skimming(["free_flow_time", "distance"])
    assigclass2 = type(assigclass)("car", car_graph, matrix)
    assignment.set_classes([assigclass2])
    assignment.set_algorithm("bfw")
    assignment.execute()
    bfw25_rgap = assignment.assignment.rgap
    correl = np.corrcoef(assigclass2.results.total_link_loads, results.volume)[0, 1]
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

    num_cores = assignment.cores
    log_ = Path(proj.path_to_file).parent / "aequilibrae.log"
    assert isfile(log_)

    with open(log_, "r", encoding="utf-8") as file:
        file_text = file.read()

    tc_spec = "INFO ; Traffic Class specification"
    assert file_text.count(tc_spec) > 1

    tc_graph = (
        "INFO ; {'car': {'Graph': \"{'Mode': 'c', 'Block through centroids': False, "
        "'Number of centroids': 24, 'Links': 76, 'Nodes': 24}\","
    )

    assert file_text.count(tc_graph) > 1

    tc_matrix = "'Number of centroids': 24, 'Matrix cores': ['matrix'], 'Matrix totals': {'matrix': 360600.0}}\"}}"
    assert file_text.count(tc_matrix) > 1

    assig_1 = (
        "INFO ; {{'VDF parameters': {{'alpha': 'b', 'beta': 'power'}}, "
        "'VDF function': 'bpr', 'Number of cores': {}, 'Capacity field': 'capacity', "
        "'Time field': 'free_flow_time', 'Algorithm': 'msa', 'Maximum iterations': 10, "
        "'Target RGAP': 0.0001, 'Line search': 'trapezoidal', "
        "'BFW conjugacy': 'approximate'}}"
    ).format(num_cores)
    assert assig_1 in file_text

    assig_2 = (
        "INFO ; {{'VDF parameters': {{'alpha': 'b', 'beta': 'power'}}, "
        "'VDF function': 'bpr', 'Number of cores': {}, 'Capacity field': 'capacity', "
        "'Time field': 'free_flow_time', 'Algorithm': 'msa', 'Maximum iterations': 500, "
        "'Target RGAP': 0.001, 'Line search': 'trapezoidal', "
        "'BFW conjugacy': 'approximate'}}"
    ).format(num_cores)
    assert assig_2 in file_text

    assig_3 = (
        "INFO ; {{'VDF parameters': {{'alpha': 'b', 'beta': 'power'}}, "
        "'VDF function': 'bpr', 'Number of cores': {}, 'Capacity field': 'capacity', "
        "'Time field': 'free_flow_time', 'Algorithm': 'frank-wolfe', "
        "'Maximum iterations': 500, 'Target RGAP': 0.001, 'Line search': 'trapezoidal', "
        "'BFW conjugacy': 'approximate'}}"
    ).format(num_cores)
    assert assig_3 in file_text

    assig_4 = (
        "INFO ; {{'VDF parameters': {{'alpha': 'b', 'beta': 'power'}}, "
        "'VDF function': 'bpr', 'Number of cores': {}, 'Capacity field': 'capacity', "
        "'Time field': 'free_flow_time', 'Algorithm': 'cfw', 'Maximum iterations': 500, "
        "'Target RGAP': 0.001, 'Line search': 'trapezoidal', "
        "'BFW conjugacy': 'approximate'}}"
    ).format(num_cores)
    assert assig_4 in file_text

    assig_5 = (
        "INFO ; {{'VDF parameters': {{'alpha': 'b', 'beta': 'power'}}, "
        "'VDF function': 'bpr', 'Number of cores': {}, 'Capacity field': 'capacity', "
        "'Time field': 'free_flow_time', 'Algorithm': 'bfw', 'Maximum iterations': 500, "
        "'Target RGAP': 0.001, 'Line search': 'trapezoidal', "
        "'BFW conjugacy': 'approximate'}}"
    ).format(num_cores)
    assert assig_5 in file_text


def test_execute_no_project(project, assignment, assigclass):
    with project.db_connection as conn:
        results = pd.read_sql("select volume from links order by link_id", conn)
    project.close()
    assignment = type(assignment)()
    assignment.add_class(assigclass)
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("free_flow_time")
    assignment.max_iter = 10
    assignment.set_algorithm("msa")
    assignment.execute()

    correl = np.corrcoef(assigclass.results.total_link_loads, results.volume.values)[0, 1]
    assert 0.8 < correl

    with pytest.raises(FileNotFoundError):
        assignment.save_results("anything")


def _configure(assignment, assigclass, algorithm="bfw", max_iter=30, rgap=1e-8):
    assignment.add_class(assigclass)
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("free_flow_time")
    assignment.max_iter = max_iter
    assignment.rgap_target = rgap
    assignment.set_algorithm(algorithm)
    return assignment


def test_line_search_defaults_to_trapezoidal(assignment, assigclass):
    _configure(assignment, assigclass)

    assert assignment.line_search == "trapezoidal"
    assert assignment.assignment.line_search == "trapezoidal"
    assert assignment._config["Line search"] == "trapezoidal"


@pytest.mark.parametrize("line_search", ["exact", "trapezoidal", "EXACT"])
def test_set_line_search_propagates_to_the_running_algorithm(assignment, assigclass, line_search):
    _configure(assignment, assigclass)

    assignment.set_line_search(line_search)

    assert assignment.line_search == line_search.lower()
    # Must reach the object that actually runs, even though it was created by set_algorithm beforehand.
    assert assignment.assignment.line_search == line_search.lower()
    assert assignment._config["Line search"] == line_search.lower()


def test_set_line_search_before_set_algorithm_is_honoured(assignment, assigclass):
    assignment.set_line_search("exact")
    _configure(assignment, assigclass)

    assert assignment.assignment.line_search == "exact"


@pytest.mark.parametrize("bad", ["quadratic", "", 1, None])
def test_set_line_search_rejects_unknown_methods(assignment, bad):
    with pytest.raises(ValueError, match="Line search must be one of"):
        assignment.set_line_search(bad)


@pytest.mark.parametrize("algorithm", ["cfw", "bfw"])
def test_exact_line_search_is_not_capped_and_changes_the_steps(assignment, assigclass, algorithm):
    """The trapezoidal path caps BFW at 1/sqrt(iter); the exact path must not, and must pick different steps."""
    _configure(assignment, assigclass, algorithm=algorithm)
    assignment.set_line_search("exact")
    assignment.execute()

    exact_alphas = np.array(assignment.assignment.convergence_report["alpha"], dtype=float)
    exact_rgap = assignment.assignment.rgap

    assert np.all(np.isfinite(exact_alphas[:-1]))
    assert np.all(exact_alphas[:-1] >= 0.0) and np.all(exact_alphas[:-1] <= 1.0)
    # An exact search on a convex objective takes longer steps than the trapezoidal overestimate.
    if algorithm == "bfw":
        cap = np.array([1.0 / np.sqrt(i) for i in assignment.assignment.convergence_report["iteration"]])
        assert np.any(exact_alphas[:-1] > cap[:-1]), "exact line search should be able to exceed the BFW cap"
    assert np.isfinite(exact_rgap)


def test_bfw_conjugacy_defaults_to_approximate(assignment, assigclass):
    _configure(assignment, assigclass)

    assert assignment.bfw_conjugacy == "approximate"
    assert assignment.assignment.bfw_conjugacy == "approximate"
    assert assignment._config["BFW conjugacy"] == "approximate"


@pytest.mark.parametrize("conjugacy", ["exact", "approximate", "EXACT"])
def test_set_bfw_conjugacy_propagates_to_the_running_algorithm(assignment, assigclass, conjugacy):
    _configure(assignment, assigclass)

    assignment.set_bfw_conjugacy(conjugacy)

    assert assignment.assignment.bfw_conjugacy == conjugacy.lower()
    assert assignment._config["BFW conjugacy"] == conjugacy.lower()


def test_set_bfw_conjugacy_before_set_algorithm_is_honoured(assignment, assigclass):
    assignment.set_bfw_conjugacy("exact")
    _configure(assignment, assigclass)

    assert assignment.assignment.bfw_conjugacy == "exact"


@pytest.mark.parametrize("bad", ["biconjugate", "", 2, None])
def test_set_bfw_conjugacy_rejects_unknown_methods(assignment, bad):
    with pytest.raises(ValueError, match="BFW conjugacy must be one of"):
        assignment.set_bfw_conjugacy(bad)


@pytest.mark.parametrize("conjugacy", ["approximate", "exact"])
def test_bfw_reports_conjugacy_diagnostics(assignment, assigclass, conjugacy):
    _configure(assignment, assigclass, algorithm="bfw", max_iter=25)
    assignment.set_bfw_conjugacy(conjugacy)
    assignment.execute()

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
    if conjugacy == "exact":
        # Only iterations where the clamp did not fire are required to be biconjugate.
        clamped = np.array([v for v in report["bfw_clamped"] if v is not None], dtype=float)
        interior = np.isfinite(clamped) & (clamped == 0.0)
        if interior.any():
            all_residual = np.array(report["conjugacy_prev2"], dtype=float)
            assert np.all(np.abs(all_residual[interior]) < 1e-8)


def test_cfw_does_not_get_conjugacy_columns(assignment, assigclass):
    _configure(assignment, assigclass, algorithm="cfw")

    assert "conjugacy_prev" not in assignment.assignment.convergence_report
