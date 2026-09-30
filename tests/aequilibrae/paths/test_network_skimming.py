from os.path import isfile, join

import numpy as np
import pandas as pd
import pytest

from aequilibrae.paths.graph import Graph
from aequilibrae.paths.network_skimming import NetworkSkimming
from aequilibrae.paths.results import PathResults, SkimResults


def test_network_skimming(sioux_falls_example):
    network = sioux_falls_example.network
    project = sioux_falls_example
    proj_dir = sioux_falls_example.project_base_path

    network.build_graphs()
    graph = network.graphs["c"]
    graph.set_graph(cost_field="distance")
    graph.set_skimming("distance")
    graph.set_blocked_centroid_flows(False)

    # skimming results
    res = SkimResults()
    res.prepare(graph)
    skm = NetworkSkimming(graph)
    skm.execute()

    tot = np.nanmax(skm.results.skims.distance[:, :])
    assert tot <= np.sum(graph.cost), "Skimming was not successful. At least one np.inf returned."
    assert not skm.report, f"Skimming returned an error: {skm.report}"

    fn = "test_Skimming"
    skm.save_to_project(fn, format="omx")
    matrix_dir = join(proj_dir, "matrices")

    assert isfile(join(matrix_dir, f"{fn}.omx")), "Did not save project to project"

    matrices = project.matrices
    mat = matrices.get(fn)
    assert mat.name == fn, "Matrix record name saved wrong"
    assert mat.file_name == f"{fn}.omx", "matrix file_name saved wrong"
    assert mat.cores == 1, "matrix saved number of matrix cores wrong"
    assert mat.procedure == "Network skimming", "Matrix saved wrong procedure name"
    assert mat.procedure_id == skm.procedure_id, "Procedure ID saved wrong"
    assert mat.timestamp == skm.procedure_date, "Procedure ID saved wrong"


@pytest.mark.parametrize("cores", [1, 2])
@pytest.mark.parametrize("turn_penalty", [None, 0.5, 4.0])
@pytest.mark.parametrize("skim_fields", [["time", "distance"], ["distance", "time"]])
def test_network_skimming_uses_routing_cost_and_turn_paths(cores, turn_penalty, skim_fields):
    graph = Graph()
    graph.network = pd.DataFrame(
        {
            "link_id": [71, 12, 55, 24],
            "a_node": [10, 20, 10, 30],
            "b_node": [20, 40, 30, 40],
            "direction": [1, 1, 1, 1],
            "time": [1.0, 1.0, 2.0, 2.0],
            "distance": [3.0, 4.0, 5.0, 6.0],
        }
    )
    graph.prepare_graph(np.array([10, 20, 30, 40]), remove_dead_ends=False)
    graph.set_blocked_centroid_flows(False)
    graph.set_graph("time")
    graph.set_skimming(skim_fields)
    if turn_penalty is not None:
        graph.set_turn_restrictions(
            pd.DataFrame({"from_node": [10], "via_node": [20], "to_node": [40], "penalty": [turn_penalty]})
        )

    skim = NetworkSkimming(graph)
    skim.set_cores(cores)
    skim.results.set_heap("pairing")
    skim.execute()
    assert skim.report == ["Centroid 40 has no outgoing edges"]
    assert skim.results.skims.names == skim_fields
    np.testing.assert_array_equal(skim.results.skims.index, graph.centroids)
    path = PathResults(graph, 10, 40)
    assert skim.results.skims.matrix["time"][0, 3] == path.milepost[-1]
    expected_distance = 11.0 if turn_penalty == 4.0 else 7.0
    assert skim.results.skims.matrix["distance"][0, 3] == expected_distance
    assert np.isinf(skim.results.skims.matrix["time"][3, 3])
    assert np.isinf(skim.results.skims.matrix["time"][3, 0])
    assert np.isinf(skim.results.skims.matrix["distance"][3, 0])
    if turn_penalty == 0.5:
        assert skim.results.skims.matrix["time"][0, 3] == 2.5

    skim.execute()
    assert skim.results.skims.matrix["time"][0, 3] == path.milepost[-1]


def test_network_skimming_no_project(sioux_falls_example):
    network = sioux_falls_example.network

    network.build_graphs()
    graph = network.graphs["c"]
    graph.set_graph(cost_field="distance")
    graph.set_skimming("distance")
    graph.set_blocked_centroid_flows(False)

    # skimming results
    skm = NetworkSkimming(graph)
    skm.execute()

    tot = np.nanmax(skm.results.skims.distance[:, :])
    assert tot <= np.sum(graph.cost), "Skimming was not successful. At least one np.inf returned."
    assert not skm.report, f"Skimming returned an error: {skm.report}"
