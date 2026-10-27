from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from aequilibrae.paths.cython.outputs import SkimmingOutputs

from aequilibrae.paths.graph import Graph
from aequilibrae.paths.network_skimming import NetworkSkimming
from aequilibrae.paths.results import PathResults, SkimResults


def test_network_skimming(sioux_falls_example):
    network = sioux_falls_example.network
    graph = network.build_graph(mode="c", cost="distance", skimming=["distance"], block_centroid_flows=False)

    # skimming results
    res = SkimResults()
    res.prepare(graph)
    skm = NetworkSkimming(graph)
    skm.execute()

    tot = np.nanmax(skm.results.skims["distance"])
    assert tot <= np.sum(graph.cost), "Skimming was not successful. At least one np.inf returned."
    assert not skm.report, f"Skimming returned an error: {skm.report}"

    matrices = sioux_falls_example.matrices
    record = matrices.create(
        "test_Skimming",
        "test_Skimming.omx",
        skm.results.skims,
        procedure="Network skimming",
        procedure_id=skm.procedure_id,
        timestamp=skm.procedure_date,
    )
    assert record.cores == 1
    assert record.procedure_id == skm.procedure_id
    loaded = matrices.get_matrix(record.name)
    np.testing.assert_array_equal(loaded["distance"], skm.results.skims["distance"])


def test_prepare_shares_existing_output():
    graph = SimpleNamespace(
        cost_field="time", num_zones=2, centroids=np.array([30, 10]), skim_fields=["time", "distance"]
    )
    output = SkimmingOutputs(2, 2, ["distance", "time"])
    results = SkimResults()
    results.prepare(graph, output=output)
    assert set(results.skims) == set(graph.skim_fields)
    for name in graph.skim_fields:
        assert np.shares_memory(results.skims[name], output.skims)


def test_empty_network_skimming():
    graph = SimpleNamespace(
        cost_field="time", num_zones=0, centroids=np.array([], dtype=np.int64), skim_fields=["time"]
    )
    skim = NetworkSkimming(graph)
    skim.execute()
    assert skim.results.skims["time"].shape == (0, 0)
    assert not skim.report


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
    assert set(skim.results.skims) == set(skim_fields)
    np.testing.assert_array_equal(skim.results.skims.index, graph.centroids)
    path = PathResults(graph, 10, 40)
    assert skim.results.skims["time"][0, 3] == path.milepost[-1]
    expected_distance = 11.0 if turn_penalty == 4.0 else 7.0
    assert skim.results.skims["distance"][0, 3] == expected_distance
    assert np.isinf(skim.results.skims["time"][3, 3])
    assert np.isinf(skim.results.skims["time"][3, 0])
    assert np.isinf(skim.results.skims["distance"][3, 0])
    if turn_penalty == 0.5:
        assert skim.results.skims["time"][0, 3] == 2.5

    skim.execute()
    assert skim.results.skims["time"][0, 3] == path.milepost[-1]


def test_network_skimming_no_project(sioux_falls_example):
    network = sioux_falls_example.network

    graph = network.build_graph(mode="c", cost="distance", skimming=["distance"], block_centroid_flows=False)

    # skimming results
    skm = NetworkSkimming(graph)
    skm.execute()

    tot = np.nanmax(skm.results.skims["distance"])
    assert tot <= np.sum(graph.cost), "Skimming was not successful. At least one np.inf returned."
    assert not skm.report, f"Skimming returned an error: {skm.report}"
