import numpy as np
import pandas as pd
import pytest

from aequilibrae.routing import Network
from aequilibrae.routing.connectivity_analysis import disconnected_analysis


def test_connectivity_analysis(sioux_falls_example):
    network = sioux_falls_example.road
    graph = network.build_graph(mode="c", cost="distance", block_centroid_flows=False)

    disconnected = graph.disconnected_nodes()
    assert disconnected.shape[0] == 0

    source = network.build_network()
    costs = source.field_values("distance").copy()
    costs[np.isin(source.link_ids, [5, 12, 19])] = np.inf  # Disconnect nodes 1, 2, and 6 from the rest of the graph
    graph.set_costs(costs)
    disconnected = graph.disconnected_nodes()
    assert disconnected.shape[0] == 3


@pytest.mark.parametrize("blocked", [False, True])
def test_centroid_endpoints_and_disconnected_islands(blocked):
    network = Network(
        pd.DataFrame(
            {
                "link_id": np.arange(1, 8),
                "a_node": [1, 2, 3, 4, 4, 6, 8],
                "b_node": [2, 3, 4, 2, 5, 7, 9],
                "direction": 0,
                "time": 1.0,
            }
        ),
        centroids=[1, 5, 8, 9, 10],
    )
    graph = network.build_graph("c", "time", block_centroid_flows=blocked)
    # Neither terminal connector disconnects the road core. Separate road and
    # centroid-only islands, and a retained isolated centroid, remain disconnected.
    assert set(graph.disconnected_nodes()) == {6, 7, 8, 9, 10}

    graph.set_costs(np.full(network.mapping.link_count, np.inf))
    np.testing.assert_array_equal(graph.disconnected_nodes(), graph.all_nodes[1:])


@pytest.mark.parametrize("blocked", [False, True])
def test_centroid_cannot_bridge_road_components_when_blocked(blocked):
    network = Network(
        pd.DataFrame(
            {
                "link_id": np.arange(1, 7),
                "a_node": [1, 2, 3, 3, 4, 5],
                "b_node": [2, 3, 1, 4, 5, 6],
                "direction": 0,
                "time": 1.0,
            }
        ),
        centroids=[4],
    )
    graph = network.build_graph("c", "time", block_centroid_flows=blocked)
    assert set(graph.disconnected_nodes()) == ({5, 6} if blocked else set())


@pytest.mark.parametrize("blocked", [False, True])
def test_one_way_endpoint_access_is_disconnected(blocked):
    network = Network(
        pd.DataFrame(
            {
                "link_id": np.arange(1, 6),
                "a_node": [1, 2, 3, 4, 4],
                "b_node": [2, 3, 4, 2, 5],
                "direction": [1, 0, 0, 0, 1],
                "time": 1.0,
            }
        ),
        centroids=[1, 5],
    )
    graph = network.build_graph("c", "time", block_centroid_flows=blocked)
    assert set(graph.disconnected_nodes()) == {1, 5}


@pytest.mark.parametrize("blocked", [False, True])
def test_no_transitions_preserves_centroid_blocking(blocked):
    graph = Network(
        pd.DataFrame({"link_id": [1, 2], "a_node": [1, 2], "b_node": [2, 3], "direction": 0, "time": 1.0}),
        centroids=[1, 2, 3, 4],
    ).build_graph("c", "time", block_centroid_flows=blocked)
    assert set(graph.disconnected_nodes()) == ({3, 4} if blocked else {4})


@pytest.mark.parametrize("blocked", [False, True])
def test_connectivity_uses_dense_indices_with_sparse_external_ids(blocked):
    maximum = np.iinfo(np.uint64).max
    # The final node is retained even though it has no active endpoint.
    nodes = np.array([0, 2**63, maximum], dtype=np.uint64)
    result = disconnected_analysis(np.array([0, 1]), np.array([1, 0]), nodes, blocked, nodes[:0])
    np.testing.assert_array_equal(result, nodes[2:])

    result = disconnected_analysis(np.array([], dtype=int), np.array([], dtype=int), nodes, blocked, nodes[:0])
    np.testing.assert_array_equal(result, nodes[1:])

    result = disconnected_analysis(np.array([], dtype=int), np.array([], dtype=int), nodes[:0], blocked, nodes[:0])
    assert result.size == 0
