import numpy as np
import pandas as pd
import pytest
from aequilibrae.paths.cython.outputs import SkimmingOutputs

from aequilibrae.paths import Network
from aequilibrae.paths.network_skimming import NetworkSkimming
from aequilibrae.paths.results import SkimResults


def test_network_skimming(sioux_falls_example):
    network = sioux_falls_example.network.build()
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
    file_name = "test_Skimming.omx"
    matrices.save_matrix(
        file_name,
        skm.results.skims,
        procedure="Network skimming",
        procedure_id=skm.procedure_id,
        timestamp=skm.procedure_date,
    )
    assert matrices.cores_on_disk(file_name) == 1
    assert matrices.get_available_matrices_and_metadata()[file_name]["procedure_id"] == skm.procedure_id
    with matrices.get_matrix(file_name) as loaded:
        np.testing.assert_array_equal(loaded["distance"], skm.results.skims["distance"])


def test_prepare_shares_existing_output():
    graph = Network(
        pd.DataFrame(
            {"link_id": [1], "a_node": [10], "b_node": [30], "direction": [0], "time": [1.0], "distance": [2.0]}
        ),
        centroids=np.array([30, 10]),
    ).build_graph("c", cost="time", skimming=["time", "distance"])
    output = SkimmingOutputs(2, 2, ["distance", "time"])
    results = SkimResults()
    results.prepare(graph, output=output)
    assert set(results.skims) == set(graph.skim_fields)
    for name in graph.skim_fields:
        assert np.shares_memory(results.skims[name], output.skims)


def test_empty_network_skimming():
    graph = Network(
        pd.DataFrame({"link_id": [1], "a_node": [10], "b_node": [30], "direction": [0], "time": [1.0]})
    ).build_graph("c", cost="time", skimming=["time"])
    skim = NetworkSkimming(graph)
    skim.execute()
    assert skim.results.skims["time"].shape == (0, 0)
    assert not skim.report


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("centroids", [[10, 20, 30, 40], [40, 10]])
@pytest.mark.parametrize("cores", [1, 2])
@pytest.mark.parametrize("turn_penalty", [None, 0.5, 4.0])
@pytest.mark.parametrize("skim_fields", [["time", "distance"], ["distance", "time"]])
def test_network_skimming_uses_routing_cost_and_turn_paths(compact, centroids, cores, turn_penalty, skim_fields):
    links = pd.DataFrame(
        {
            "link_id": [71, 12, 55, 24],
            "a_node": [10, 20, 10, 30],
            "b_node": [20, 40, 30, 40],
            "direction": [1, 1, 1, 1],
            "time": [1.0, 1.0, 2.0, 2.0],
            "distance": [3.0, 4.0, 5.0, 6.0],
        }
    )
    turns = (
        None
        if turn_penalty is None
        else pd.DataFrame({"from_node": [10], "via_node": [20], "to_node": [40], "penalty": [turn_penalty]})
    )
    graph = Network(links, centroids=np.array(centroids), turn_restrictions=turns).build_graph(
        "c",
        cost="time",
        skimming=skim_fields,
        compact=compact,
        block_centroid_flows=False,
    )

    skim = NetworkSkimming(graph)
    skim.set_cores(cores)
    skim.results.set_heap("pairing")
    skim.execute()
    assert skim.report == ["Centroid 40 has no outgoing edges"]
    assert set(skim.results.skims) == set(skim_fields)
    np.testing.assert_array_equal(skim.results.skims.index, graph.centroids)
    origin, destination = centroids.index(10), centroids.index(40)
    expected_time = min(2.0 + (turn_penalty or 0.0), 4.0)
    assert skim.results.skims["time"][origin, destination] == expected_time
    expected_distance = 11.0 if turn_penalty == 4.0 else 7.0
    assert skim.results.skims["distance"][origin, destination] == expected_distance
    assert np.isinf(skim.results.skims["time"][destination, destination])
    assert np.isinf(skim.results.skims["time"][destination, origin])
    assert np.isinf(skim.results.skims["distance"][destination, origin])

    graph.set_costs([3.0, 3.0, 1.0, 1.0])
    skim.execute()
    assert skim.results.skims["time"][origin, destination] == 2.0
    assert skim.results.skims["distance"][origin, destination] == 11.0


def test_network_skimming_no_project(sioux_falls_example):
    network = sioux_falls_example.network.build()

    graph = network.build_graph(mode="c", cost="distance", skimming=["distance"], block_centroid_flows=False)

    # skimming results
    skm = graph.compute_skims(cores=1)
    assert skm.cores == 1

    tot = np.nanmax(skm.results.skims["distance"])
    assert tot <= np.sum(graph.cost), "Skimming was not successful. At least one np.inf returned."
    assert not skm.report, f"Skimming returned an error: {skm.report}"
