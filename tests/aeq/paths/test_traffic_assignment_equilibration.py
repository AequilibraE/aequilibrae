from os.path import isfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aequilibrae import TrafficAssignment, TrafficClass, Graph


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
    bfw25_iters = assignment.assignment.iter
    correl = np.corrcoef(assigclass2.results.total_link_loads, results.volume)[0, 1]
    assert 0.999 < correl

    assert msa25_rgap < msa10_rgap
    assert fw25_rgap < msa25_rgap
    assert cfw25_rgap < assignment.rgap_target
    assert bfw25_rgap < assignment.rgap_target
    assert cfw25_iters < fw25_iters
    assert bfw25_iters < cfw25_iters

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

    tc_graph = "INFO ; {'car': {'Graph': \"{'Mode': 'c', 'Block through centroids': False, 'Number of centroids': 24, 'Links': 76, 'Nodes': 24}\","
    assert file_text.count(tc_graph) > 1

    tc_matrix = "'Number of centroids': 24, 'Matrix cores': ['matrix'], 'Matrix totals': {'matrix': 360600.0}}\"}}"
    assert file_text.count(tc_matrix) > 1

    assig_1 = "INFO ; {{'VDF parameters': {{'alpha': 'b', 'beta': 'power'}}, 'VDF function': 'bpr', 'Number of cores': {}, 'Capacity field': 'capacity', 'Time field': 'free_flow_time', 'Algorithm': 'msa', 'Maximum iterations': 10, 'Target RGAP': 0.0001}}".format(
        num_cores
    )
    assert assig_1 in file_text

    assig_2 = "INFO ; {{'VDF parameters': {{'alpha': 'b', 'beta': 'power'}}, 'VDF function': 'bpr', 'Number of cores': {}, 'Capacity field': 'capacity', 'Time field': 'free_flow_time', 'Algorithm': 'msa', 'Maximum iterations': 500, 'Target RGAP': 0.001}}".format(
        num_cores
    )
    assert assig_2 in file_text

    assig_3 = "INFO ; {{'VDF parameters': {{'alpha': 'b', 'beta': 'power'}}, 'VDF function': 'bpr', 'Number of cores': {}, 'Capacity field': 'capacity', 'Time field': 'free_flow_time', 'Algorithm': 'frank-wolfe', 'Maximum iterations': 500, 'Target RGAP': 0.001}}".format(
        num_cores
    )
    assert assig_3 in file_text

    assig_4 = "INFO ; {{'VDF parameters': {{'alpha': 'b', 'beta': 'power'}}, 'VDF function': 'bpr', 'Number of cores': {}, 'Capacity field': 'capacity', 'Time field': 'free_flow_time', 'Algorithm': 'cfw', 'Maximum iterations': 500, 'Target RGAP': 0.001}}".format(
        num_cores
    )
    assert assig_4 in file_text

    assig_5 = "INFO ; {{'VDF parameters': {{'alpha': 'b', 'beta': 'power'}}, 'VDF function': 'bpr', 'Number of cores': {}, 'Capacity field': 'capacity', 'Time field': 'free_flow_time', 'Algorithm': 'bfw', 'Maximum iterations': 500, 'Target RGAP': 0.001}}".format(
        num_cores
    )
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
