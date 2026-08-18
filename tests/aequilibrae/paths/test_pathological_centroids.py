"""Pathological tests for centroid-only path blocking semantics."""

import networkx as nx
import pytest

from .pathological_components import (
    make_intermediate_centroid_component,
    make_shared_junction_centroid_component,
)
from .pathological_network import OraclePath, PathologicalNetwork


@pytest.fixture
def shared_junction_centroid_network() -> PathologicalNetwork:
    """Centroid connectors meet legally at one ordinary shared junction."""
    return PathologicalNetwork.compose(make_shared_junction_centroid_component())


@pytest.fixture
def intermediate_centroid_network() -> PathologicalNetwork:
    """A cheap route crosses a centroid while a unique bypass avoids it."""
    return PathologicalNetwork.compose(make_intermediate_centroid_component())


def _assert_unique_oracle_matches_aequilibrae(
    network: PathologicalNetwork,
    origin_name: str,
    destination_name: str,
    *,
    block_centroid_flows: bool,
) -> OraclePath:
    origin = network.node_id(origin_name)
    destination = network.node_id(destination_name)
    state_graph = network.oracle_state_graph(
        origin,
        destination,
        block_centroid_flows=block_centroid_flows,
    )
    shortest_paths = list(
        nx.all_shortest_paths(
            state_graph,
            ("source", origin),
            ("sink", destination),
            weight="weight",
        )
    )
    assert len(shortest_paths) == 1
    expected = network.oracle_path(
        origin,
        destination,
        block_centroid_flows=block_centroid_flows,
    )

    graph = network.build_graph(block_centroid_flows=block_centroid_flows)
    actual = graph.compute_path(origin, destination)

    assert actual.path is not None
    assert network.result_directed_links(actual) == expected.directed_links
    assert network.result_generalized_cost(
        actual,
        block_centroid_flows=block_centroid_flows,
    ) == pytest.approx(expected.cost)
    return expected


@pytest.mark.parametrize("block_centroid_flows", [False, True])
def test_shared_ordinary_junction_remains_traversable(
    shared_junction_centroid_network: PathologicalNetwork,
    block_centroid_flows: bool,
) -> None:
    """Blocking never mistakes an ordinary shared junction for a centroid."""
    origin = shared_junction_centroid_network.node_id("shared_junction_centroid:origin")
    junction = shared_junction_centroid_network.node_id("shared_junction_centroid:junction")
    directed_graph = shared_junction_centroid_network.networkx_graph()
    assert directed_graph.out_degree(origin) == 2
    assert junction not in shared_junction_centroid_network.centroids

    expected = _assert_unique_oracle_matches_aequilibrae(
        shared_junction_centroid_network,
        "shared_junction_centroid:origin",
        "shared_junction_centroid:destination",
        block_centroid_flows=block_centroid_flows,
    )

    assert expected.nodes == tuple(
        shared_junction_centroid_network.node_id(f"shared_junction_centroid:{name}")
        for name in ("origin", "junction", "destination")
    )


@pytest.mark.parametrize(
    ("block_centroid_flows", "expected_names"),
    [
        (False, ("origin", "left", "middle", "right", "destination")),
        (True, ("origin", "left", "right", "destination")),
    ],
)
def test_only_true_intermediate_centroid_is_blocked(
    intermediate_centroid_network: PathologicalNetwork,
    block_centroid_flows: bool,
    expected_names: tuple[str, ...],
) -> None:
    """Blocking removes traversal through middle and selects its unique bypass."""
    expected = _assert_unique_oracle_matches_aequilibrae(
        intermediate_centroid_network,
        "intermediate_centroid:origin",
        "intermediate_centroid:destination",
        block_centroid_flows=block_centroid_flows,
    )

    assert expected.nodes == tuple(
        intermediate_centroid_network.node_id(f"intermediate_centroid:{name}") for name in expected_names
    )


@pytest.mark.parametrize(
    ("origin_name", "destination_name", "expected_names"),
    [
        ("origin", "middle", ("origin", "left", "middle")),
        ("middle", "destination", ("middle", "right", "destination")),
    ],
)
def test_blocked_centroid_remains_legal_as_path_endpoint(
    intermediate_centroid_network: PathologicalNetwork,
    origin_name: str,
    destination_name: str,
    expected_names: tuple[str, ...],
) -> None:
    """Centroid blocking applies to transitions, not endpoint access."""
    expected = _assert_unique_oracle_matches_aequilibrae(
        intermediate_centroid_network,
        f"intermediate_centroid:{origin_name}",
        f"intermediate_centroid:{destination_name}",
        block_centroid_flows=True,
    )

    assert expected.nodes == tuple(
        intermediate_centroid_network.node_id(f"intermediate_centroid:{name}") for name in expected_names
    )


@pytest.mark.parametrize("block_centroid_flows", [False, True])
def test_one_connector_centroid_and_reverse_oriented_link_are_honored(
    intermediate_centroid_network: PathologicalNetwork,
    block_centroid_flows: bool,
) -> None:
    """A leaf centroid remains reachable through its sole reverse-oriented connector."""
    origin = intermediate_centroid_network.node_id("intermediate_centroid:origin")
    left = intermediate_centroid_network.node_id("intermediate_centroid:left")
    middle = intermediate_centroid_network.node_id("intermediate_centroid:middle")
    single = intermediate_centroid_network.node_id("intermediate_centroid:single")
    directed_graph = intermediate_centroid_network.networkx_graph()
    assert directed_graph.has_edge(origin, left) and directed_graph.has_edge(left, origin)
    assert directed_graph.has_edge(left, middle) and not directed_graph.has_edge(middle, left)
    assert directed_graph.has_edge(left, single) and not directed_graph.has_edge(single, left)
    assert directed_graph.degree(single) == 1

    expected = _assert_unique_oracle_matches_aequilibrae(
        intermediate_centroid_network,
        "intermediate_centroid:origin",
        "intermediate_centroid:single",
        block_centroid_flows=block_centroid_flows,
    )

    assert expected.nodes == tuple(
        intermediate_centroid_network.node_id(f"intermediate_centroid:{name}") for name in ("origin", "left", "single")
    )
    assert expected.directed_links[-1] == (
        intermediate_centroid_network.link_id("intermediate_centroid:single_connector"),
        -1,
    )
