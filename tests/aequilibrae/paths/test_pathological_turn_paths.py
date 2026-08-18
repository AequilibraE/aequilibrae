"""Pathological shortest-path contracts for link directions and turn controls."""

import networkx as nx
import pytest

from .pathological_components import (
    make_disconnected_component,
    make_duplicate_uturn_control_component,
    make_equal_parallel_component,
    make_prohibited_parallel_component,
    make_prohibited_uturn_component,
    make_selected_finite_turn_component,
    make_uturn_component,
)
from .pathological_network import DirectedArc, OraclePath, PathologicalNetwork


@pytest.fixture
def prohibited_parallel_network() -> PathologicalNetwork:
    """Parallel incoming arcs share one prohibited node movement and one obvious bypass."""
    return PathologicalNetwork.compose(make_prohibited_parallel_component())


@pytest.fixture
def unrestricted_parallel_network() -> PathologicalNetwork:
    """Two equal parallel incoming arc states remain valid without prescribing tie identity."""
    return PathologicalNetwork.compose(make_equal_parallel_component())


@pytest.fixture
def uturn_network() -> PathologicalNetwork:
    """A finite branch U-turn can avoid a prohibited direct movement when globally allowed."""
    return PathologicalNetwork.compose(make_uturn_component())


@pytest.fixture
def prohibited_uturn_network() -> PathologicalNetwork:
    """A prohibited branch U-turn leaves only the component's expensive escape route."""
    return PathologicalNetwork.compose(make_prohibited_uturn_component())


@pytest.fixture
def duplicate_uturn_control_network() -> PathologicalNetwork:
    """Finite and prohibited duplicates expose prohibition-dominant precedence."""
    return PathologicalNetwork.compose(make_duplicate_uturn_control_component())


@pytest.fixture
def disconnected_network() -> PathologicalNetwork:
    """A destination island is disconnected from the reachable two-node fragment."""
    return PathologicalNetwork.compose(make_disconnected_component())


@pytest.fixture
def selected_finite_turn_network() -> PathologicalNetwork:
    """A prohibited detour forces selection of the finite controlled movement."""
    return PathologicalNetwork.compose(make_selected_finite_turn_component())


def _result_cost_breakdown(network: PathologicalNetwork, result) -> tuple[float, float]:
    directed_links = network.result_directed_links(result)
    arc_lookup = {arc.directed_link: arc for arc in network.directed_arcs()}
    arcs = tuple(arc_lookup[directed_link] for directed_link in directed_links)
    link_cost = sum(arc.cost for arc in arcs)
    restrictions = network.turn_lookup()
    turn_cost = sum(
        restrictions.get((incoming.tail, incoming.head, outgoing.head), 0.0)
        for incoming, outgoing in zip(arcs[:-1], arcs[1:], strict=True)
    )
    return float(link_cost), float(turn_cost)


def _assert_unique_path_result(
    network: PathologicalNetwork,
    expected: OraclePath,
    result,
    *,
    allow_path_uturns: bool = False,
) -> None:
    assert result.path is not None
    assert network.result_directed_links(result) == expected.directed_links
    assert network.result_generalized_cost(result, allow_path_uturns=allow_path_uturns) == pytest.approx(expected.cost)


@pytest.mark.parametrize(
    ("origin_name", "destination_name"),
    (
        ("west", "centre"),
        ("centre", "west"),
        ("centre", "east"),
        ("east", "centre"),
        ("north", "centre"),
        ("centre", "north"),
    ),
)
def test_mixed_link_directions_expand_to_networkx_reachability(
    mixed_direction_network: PathologicalNetwork,
    origin_name: str,
    destination_name: str,
):
    """AB-only, BA-only, and bidirectional links expose exactly their NetworkX-directed paths."""
    component_name = mixed_direction_network.components[0].name
    origin = mixed_direction_network.node_id(f"{component_name}:{origin_name}")
    destination = mixed_direction_network.node_id(f"{component_name}:{destination_name}")
    try:
        expected = mixed_direction_network.oracle_path(origin, destination)
    except nx.NetworkXNoPath:
        expected = None

    graph = mixed_direction_network.build_graph()
    result = graph.compute_path(origin, destination)

    if expected is None:
        assert result.path is None
    else:
        _assert_unique_path_result(mixed_direction_network, expected, result)


def test_finite_penalty_changes_route_and_generalized_cost(finite_turn_network: PathologicalNetwork):
    """The finite movement cost makes NetworkX choose the longer uncontrolled branch."""
    origin = finite_turn_network.node_id("finite_turn:origin")
    destination = finite_turn_network.node_id("finite_turn:destination")
    expected = finite_turn_network.oracle_path(origin, destination)

    graph = finite_turn_network.build_graph()
    result = graph.compute_path(origin, destination)

    _assert_unique_path_result(finite_turn_network, expected, result)
    actual_link_cost, actual_turn_cost = _result_cost_breakdown(finite_turn_network, result)
    assert actual_link_cost == pytest.approx(expected.link_cost)
    assert actual_turn_cost == pytest.approx(expected.turn_cost)


def test_terminal_milepost_includes_a_selected_finite_penalty(
    selected_finite_turn_network: PathologicalNetwork,
):
    """A selected controlled movement reports NetworkX link plus turn cost at its terminal milepost."""
    origin = selected_finite_turn_network.node_id("selected_finite_turn:origin")
    destination = selected_finite_turn_network.node_id("selected_finite_turn:destination")
    expected = selected_finite_turn_network.oracle_path(origin, destination)

    graph = selected_finite_turn_network.build_graph()
    result = graph.compute_path(origin, destination)

    _assert_unique_path_result(selected_finite_turn_network, expected, result)
    actual_link_cost, actual_turn_cost = _result_cost_breakdown(selected_finite_turn_network, result)
    assert actual_link_cost == pytest.approx(expected.link_cost)
    assert actual_turn_cost == pytest.approx(expected.turn_cost)
    assert result.milepost[-1] == pytest.approx(expected.cost)


def test_prohibited_movement_applies_to_every_parallel_incoming_arc(
    prohibited_parallel_network: PathologicalNetwork,
):
    """One node movement prohibition removes both parallel short states, forcing the NetworkX bypass."""
    origin = prohibited_parallel_network.node_id("prohibited_parallel:origin")
    destination = prohibited_parallel_network.node_id("prohibited_parallel:destination")
    expected = prohibited_parallel_network.oracle_path(origin, destination)

    graph = prohibited_parallel_network.build_graph()
    result = graph.compute_path(origin, destination)

    _assert_unique_path_result(prohibited_parallel_network, expected, result)


def test_equal_cost_parallel_states_do_not_prescribe_tie_identity(
    unrestricted_parallel_network: PathologicalNetwork,
):
    """Either NetworkX-equal incoming arc identity is valid while route cost and continuation stay fixed."""
    origin = unrestricted_parallel_network.node_id("equal_parallel:origin")
    destination = unrestricted_parallel_network.node_id("equal_parallel:destination")
    state_graph = unrestricted_parallel_network.oracle_state_graph(origin, destination)
    oracle_states = nx.all_shortest_paths(
        state_graph,
        ("source", origin),
        ("sink", destination),
        weight="weight",
    )
    expected_paths = {
        tuple(state.directed_link for state in states if isinstance(state, DirectedArc)) for states in oracle_states
    }
    expected_cost = nx.shortest_path_length(
        state_graph,
        ("source", origin),
        ("sink", destination),
        weight="weight",
    )

    graph = unrestricted_parallel_network.build_graph()
    result = graph.compute_path(origin, destination)

    assert len(expected_paths) == 2
    assert result.path is not None
    assert unrestricted_parallel_network.result_directed_links(result) in expected_paths
    assert unrestricted_parallel_network.result_generalized_cost(result) == pytest.approx(expected_cost)


def test_disconnected_destination_has_no_path(disconnected_network: PathologicalNetwork):
    """NetworkX and AequilibraE report no route from the reachable fragment to the disconnected island."""
    origin = disconnected_network.node_id("disconnected:origin")
    destination = disconnected_network.node_id("disconnected:destination")
    state_graph = disconnected_network.oracle_state_graph(origin, destination)
    expected_exists = nx.has_path(state_graph, ("source", origin), ("sink", destination))

    graph = disconnected_network.build_graph()
    result = graph.compute_path(origin, destination)

    assert not expected_exists
    assert result.path is None
    assert result.path_nodes is None
    assert result.milepost is None


def test_global_path_uturn_policy_is_absolute(uturn_network: PathologicalNetwork):
    """NetworkX blocks the finite branch reversal globally when false and permits it when true."""
    origin = uturn_network.node_id("uturn:origin")
    destination = uturn_network.node_id("uturn:destination")
    expected_blocked = uturn_network.oracle_path(origin, destination, allow_path_uturns=False)
    expected_allowed = uturn_network.oracle_path(origin, destination, allow_path_uturns=True)

    blocked_graph = uturn_network.build_graph(allow_uturns_everywhere=True, allow_path_uturns=False)
    blocked_result = blocked_graph.compute_path(origin, destination)
    allowed_graph = uturn_network.build_graph(allow_uturns_everywhere=True, allow_path_uturns=True)
    allowed_result = allowed_graph.compute_path(origin, destination)

    assert expected_blocked.directed_links != expected_allowed.directed_links
    _assert_unique_path_result(uturn_network, expected_blocked, blocked_result)
    _assert_unique_path_result(uturn_network, expected_allowed, allowed_result, allow_path_uturns=True)


def test_explicit_finite_uturn_contributes_to_cost(uturn_network: PathologicalNetwork):
    """When U-turns are enabled, NetworkX's selected reversal includes its explicit finite cost."""
    origin = uturn_network.node_id("uturn:origin")
    destination = uturn_network.node_id("uturn:destination")
    expected = uturn_network.oracle_path(origin, destination, allow_path_uturns=True)

    graph = uturn_network.build_graph(allow_uturns_everywhere=True, allow_path_uturns=True)
    result = graph.compute_path(origin, destination)

    _assert_unique_path_result(uturn_network, expected, result, allow_path_uturns=True)
    actual_link_cost, actual_turn_cost = _result_cost_breakdown(uturn_network, result)
    assert actual_link_cost == pytest.approx(expected.link_cost)
    assert actual_turn_cost == pytest.approx(expected.turn_cost)
    assert result.milepost[-1] == pytest.approx(expected.cost)


def test_explicit_uturn_prohibition_applies_when_uturns_are_globally_allowed(
    prohibited_uturn_network: PathologicalNetwork,
):
    """Global permission does not override NetworkX's explicit branch-reversal prohibition."""
    origin = prohibited_uturn_network.node_id("prohibited_uturn:origin")
    destination = prohibited_uturn_network.node_id("prohibited_uturn:destination")
    expected = prohibited_uturn_network.oracle_path(origin, destination, allow_path_uturns=True)

    graph = prohibited_uturn_network.build_graph(allow_uturns_everywhere=True, allow_path_uturns=True)
    result = graph.compute_path(origin, destination)

    _assert_unique_path_result(prohibited_uturn_network, expected, result, allow_path_uturns=True)


def test_prohibition_dominates_duplicate_finite_uturn_controls(
    duplicate_uturn_control_network: PathologicalNetwork,
):
    """NetworkX keeps a duplicate U-turn prohibited despite a later finite control for the same movement."""
    origin = duplicate_uturn_control_network.node_id("duplicate_uturn_control:origin")
    destination = duplicate_uturn_control_network.node_id("duplicate_uturn_control:destination")
    expected = duplicate_uturn_control_network.oracle_path(origin, destination, allow_path_uturns=True)

    graph = duplicate_uturn_control_network.build_graph(allow_uturns_everywhere=True, allow_path_uturns=True)
    result = graph.compute_path(origin, destination)

    _assert_unique_path_result(duplicate_uturn_control_network, expected, result, allow_path_uturns=True)
