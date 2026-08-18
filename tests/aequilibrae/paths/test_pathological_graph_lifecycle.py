"""Oracle-backed lifecycle tests for a graph with a finite turn cost."""

import networkx as nx
import pytest

from .pathological_components import (
    make_all_centroid_finite_turn_component,
    make_finite_turn_no_controls_component,
    make_finite_turn_without_detour_component,
    make_reversed_finite_turn_component,
)
from .pathological_network import PathologicalNetwork


@pytest.fixture
def all_centroid_finite_turn_network() -> PathologicalNetwork:
    """The finite-turn lifecycle example with every node marked as a centroid."""
    return PathologicalNetwork.compose(make_all_centroid_finite_turn_component())


@pytest.fixture
def finite_turn_no_controls_network() -> PathologicalNetwork:
    """The lifecycle topology without movement controls."""
    return PathologicalNetwork.compose(make_finite_turn_no_controls_component())


@pytest.fixture
def finite_turn_without_detour_network() -> PathologicalNetwork:
    """The lifecycle topology without its unique detour egress."""
    return PathologicalNetwork.compose(make_finite_turn_without_detour_component())


@pytest.fixture
def reversed_finite_turn_network() -> PathologicalNetwork:
    """The lifecycle topology with physical links and movement controls reversed."""
    return PathologicalNetwork.compose(make_reversed_finite_turn_component())


def test_clear_turn_restrictions_restores_no_turn_shortest_path(
    all_centroid_finite_turn_network: PathologicalNetwork,
    finite_turn_no_controls_network: PathologicalNetwork,
) -> None:
    """Clearing controls restores the route defined by the same links without turns."""
    origin = finite_turn_no_controls_network.node_id("finite_turn_no_controls:origin")
    destination = finite_turn_no_controls_network.node_id("finite_turn_no_controls:destination")
    expected = finite_turn_no_controls_network.oracle_path(origin, destination)

    graph = all_centroid_finite_turn_network.build_graph()
    graph.clear_turn_restrictions()
    graph.set_skimming(["cost"])
    actual = graph.compute_path(origin, destination)

    assert not graph.has_turn_restrictions
    assert not graph.allow_path_uturns
    assert actual.path is not None
    assert tuple(int(node) for node in actual.path_nodes) == expected.nodes
    assert finite_turn_no_controls_network.result_directed_links(actual) == expected.directed_links
    assert finite_turn_no_controls_network.result_generalized_cost(actual) == pytest.approx(expected.cost)
    assert actual.skims[destination, 0] == pytest.approx(expected.cost)


def test_excluding_detour_link_matches_component_without_that_link(
    all_centroid_finite_turn_network: PathologicalNetwork,
    finite_turn_without_detour_network: PathologicalNetwork,
) -> None:
    """Exclusion rebuilds active movement controls against the remaining topology."""
    origin = finite_turn_without_detour_network.node_id("finite_turn_without_detour:origin")
    destination = finite_turn_without_detour_network.node_id("finite_turn_without_detour:destination")
    oracle_graph = finite_turn_without_detour_network.oracle_state_graph(origin, destination)
    expected_exists = nx.has_path(oracle_graph, ("source", origin), ("sink", destination))
    expected = finite_turn_without_detour_network.oracle_path(origin, destination) if expected_exists else None

    graph = all_centroid_finite_turn_network.build_graph()
    excluded_link = all_centroid_finite_turn_network.link_id("finite_turn_all_centroids:detour_out")
    graph.exclude_links([excluded_link])
    graph.set_skimming(["cost"])
    actual = graph.compute_path(origin, destination)

    assert len(finite_turn_without_detour_network.components[0].links) == (
        len(all_centroid_finite_turn_network.components[0].links) - 1
    )
    assert (actual.path is not None) is expected_exists
    assert graph.has_turn_restrictions
    if expected is not None:
        assert tuple(int(node) for node in actual.path_nodes) == expected.nodes
        assert finite_turn_without_detour_network.result_directed_links(actual) == expected.directed_links
        assert finite_turn_without_detour_network.result_generalized_cost(actual) == pytest.approx(expected.cost)
        assert actual.skims[destination, 0] == pytest.approx(expected.cost)


def test_reverse_matches_independently_reversed_component(
    all_centroid_finite_turn_network: PathologicalNetwork,
    reversed_finite_turn_network: PathologicalNetwork,
) -> None:
    """Reversal swaps endpoints and movement triples while retaining link directions."""
    origin = reversed_finite_turn_network.node_id("finite_turn_reversed:destination")
    destination = reversed_finite_turn_network.node_id("finite_turn_reversed:origin")
    expected = reversed_finite_turn_network.oracle_path(
        origin,
        destination,
        allow_path_uturns=True,
    )

    graph = all_centroid_finite_turn_network.build_graph(allow_path_uturns=True)
    reversed_graph = graph.reverse()
    reversed_graph.set_skimming(["cost"])
    actual = reversed_graph.compute_path(origin, destination)

    assert reversed_graph.has_turn_restrictions
    assert reversed_graph.allow_path_uturns
    assert actual.path is not None
    assert tuple(int(node) for node in actual.path_nodes) == expected.nodes
    assert reversed_finite_turn_network.result_directed_links(actual) == expected.directed_links
    assert reversed_finite_turn_network.result_generalized_cost(
        actual,
        allow_path_uturns=True,
    ) == pytest.approx(expected.cost)
    assert actual.skims[destination, 0] == pytest.approx(expected.cost)


def test_reprepare_preserves_turn_controls_with_centroids(
    all_centroid_finite_turn_network: PathologicalNetwork,
) -> None:
    """Preparing again keeps one effective control while rebuilding centroid state."""
    origin = all_centroid_finite_turn_network.node_id("finite_turn_all_centroids:origin")
    destination = all_centroid_finite_turn_network.node_id("finite_turn_all_centroids:destination")
    expected = all_centroid_finite_turn_network.oracle_path(origin, destination)

    graph = all_centroid_finite_turn_network.build_graph(block_centroid_flows=False)
    graph.prepare_graph(all_centroid_finite_turn_network.centroids, remove_dead_ends=False)
    graph.set_graph("cost")
    graph.set_blocked_centroid_flows(False)
    graph.set_skimming(["cost"])
    actual = graph.compute_path(origin, destination)

    assert {int(node) for node in all_centroid_finite_turn_network.centroids} == {
        all_centroid_finite_turn_network.node_id(f"finite_turn_all_centroids:{node.name}")
        for node in all_centroid_finite_turn_network.components[0].nodes
    }
    assert graph.has_turn_restrictions
    assert not graph.allow_path_uturns
    assert actual.path is not None
    assert tuple(int(node) for node in actual.path_nodes) == expected.nodes
    assert all_centroid_finite_turn_network.result_directed_links(actual) == expected.directed_links
    assert all_centroid_finite_turn_network.result_generalized_cost(actual) == pytest.approx(expected.cost)
    assert actual.skims[destination, 0] == pytest.approx(expected.cost)
