"""Composition coverage for all human-sized pathological network components."""

from pathlib import Path

import pytest
from shapely.geometry import LineString, Point

from .dump_pathological_networks import dump_pathological_networks
from .pathological_components import (
    BRIDGE_COST,
    make_base_pathological_components,
    make_composed_pathological_network,
    make_pathological_components,
)
from .pathological_network import PathologicalNetwork


COMPONENT_NAMES = (
    "finite_turn",
    "mixed_direction",
    "prohibited_parallel",
    "uturn",
    "disconnected",
    "compression",
    "dead_end",
    "shared_junction_centroid",
    "intermediate_centroid",
)
BRIDGE_NAMES = (
    "finite_turn_to_mixed_direction",
    "mixed_direction_to_prohibited_parallel",
    "prohibited_parallel_to_uturn",
    "uturn_to_disconnected",
    "disconnected_to_compression",
    "compression_to_dead_end",
    "dead_end_to_shared_junction_centroid",
    "shared_junction_centroid_to_intermediate_centroid",
)


@pytest.fixture
def pathological_network() -> PathologicalNetwork:
    """Nine human-sized pathologies joined in order by costly two-way bridges."""
    return make_composed_pathological_network()


def _fresh_components():
    return make_base_pathological_components()


def _assert_route_matches_oracle(
    network: PathologicalNetwork,
    component_name: str,
    origin_name: str,
    destination_name: str,
    *,
    block_centroid_flows: bool = False,
) -> None:
    origin = network.node_id(f"{component_name}:{origin_name}")
    destination = network.node_id(f"{component_name}:{destination_name}")

    expected = network.oracle_path(
        origin,
        destination,
        block_centroid_flows=block_centroid_flows,
    )
    bridge_ids = {network.link_id(f"bridge:{bridge.name}") for bridge in network.bridges}
    assert bridge_ids.isdisjoint(link_id for link_id, _direction in expected.directed_links)

    graph = network.build_graph(block_centroid_flows=block_centroid_flows)
    actual = graph.compute_path(origin, destination)
    assert actual.path is not None

    assert tuple(int(node) for node in actual.path_nodes) == expected.nodes
    assert network.result_directed_links(actual) == expected.directed_links
    assert network.result_generalized_cost(
        actual,
        block_centroid_flows=block_centroid_flows,
    ) == pytest.approx(expected.cost)


def test_composition_preserves_definitions_and_assigns_unique_ids(pathological_network):
    expected_components = _fresh_components()
    expected_purposes = {component.name: component.purpose for component in expected_components}

    assert pathological_network.components == expected_components
    assert tuple(component.name for component in pathological_network.components) == COMPONENT_NAMES
    assert pathological_network.purposes == expected_purposes
    assert all(purpose.strip() for purpose in pathological_network.purposes.values())
    assert tuple(bridge.name for bridge in pathological_network.bridges) == BRIDGE_NAMES
    total_internal_link_cost = sum(
        link.cost
        for component in pathological_network.components
        for link in component.links
    )
    assert all(
        bridge.direction == 0 and bridge.cost == BRIDGE_COST
        for bridge in pathological_network.bridges
    )
    assert all(
        bridge.cost > total_internal_link_cost
        for bridge in pathological_network.bridges
    )
    assert all(
        "disconnected:destination" not in (bridge.a, bridge.b)
        for bridge in pathological_network.bridges
    )
    assert any(
        "disconnected:reachable" in (bridge.a, bridge.b)
        for bridge in pathological_network.bridges
    )

    nodes = pathological_network.node_frame()
    links = pathological_network.link_frame()
    turns = pathological_network.turn_frame()
    assert nodes.node_id.is_unique
    assert links.link_id.is_unique
    assert turns.turn_id.is_unique
    assert (nodes.node_id > 0).all()
    assert (links.link_id > 0).all()
    assert (turns.turn_id > 0).all()


def test_composition_materializes_fresh_derived_geometries(pathological_network):
    first_nodes = pathological_network.node_frame()
    second_nodes = pathological_network.node_frame()
    first_links = pathological_network.link_frame()
    second_links = pathological_network.link_frame()
    first_turns = pathological_network.turn_frame()
    second_turns = pathological_network.turn_frame()

    assert all(isinstance(geometry, Point) for geometry in first_nodes.geometry)
    assert all(isinstance(geometry, LineString) for geometry in first_links.geometry)
    assert all(isinstance(geometry, LineString) for geometry in first_turns.geometry)
    points = first_nodes.set_index("node_id").geometry
    for link in first_links.itertuples(index=False):
        assert Point(link.geometry.coords[0]).equals(points.loc[link.a_node])
        assert Point(link.geometry.coords[-1]).equals(points.loc[link.b_node])
    for turn in first_turns.itertuples(index=False):
        assert Point(turn.geometry.coords[0]).equals(points.loc[turn.from_node])
        assert Point(turn.geometry.coords[1]).equals(points.loc[turn.via_node])
        assert Point(turn.geometry.coords[-1]).equals(points.loc[turn.to_node])
    for first, second in zip(first_nodes.geometry, second_nodes.geometry, strict=True):
        assert first.equals(second)
        assert first is not second
    for first, second in zip(first_links.geometry, second_links.geometry, strict=True):
        assert first.equals(second)
        assert first is not second
    for first, second in zip(first_turns.geometry, second_turns.geometry, strict=True):
        assert first.equals(second)
        assert first is not second

    assert all(
        "geometry" not in vars(node)
        for component in pathological_network.components
        for node in component.nodes
    )
    assert all(
        "geometry" not in vars(link)
        for component in pathological_network.components
        for link in component.links
    )
    assert all(
        "geometry" not in vars(turn)
        for component in pathological_network.components
        for turn in component.turns
    )
    assert all("geometry" not in vars(bridge) for bridge in pathological_network.bridges)


def test_embedded_finite_turn_route_matches_networkx(pathological_network):
    _assert_route_matches_oracle(pathological_network, "finite_turn", "origin", "destination")


def test_embedded_prohibited_turn_route_matches_networkx(pathological_network):
    _assert_route_matches_oracle(pathological_network, "prohibited_parallel", "origin", "destination")


def test_embedded_shared_junction_centroid_route_matches_networkx(pathological_network):
    _assert_route_matches_oracle(
        pathological_network,
        "shared_junction_centroid",
        "origin",
        "destination",
        block_centroid_flows=True,
    )


def test_embedded_intermediate_centroid_route_matches_networkx(pathological_network):
    _assert_route_matches_oracle(
        pathological_network,
        "intermediate_centroid",
        "origin",
        "destination",
        block_centroid_flows=True,
    )


def test_dump_helper_writes_components_and_composed_network(tmp_path: Path):
    output_directory = tmp_path / "pathological_networks"
    dumped = dump_pathological_networks(output_directory)

    assert set(dumped) == {component.name for component in make_pathological_components()} | {"composed"}
    for name, paths in dumped.items():
        expected_directory = output_directory / name
        expected_paths = {
            "nodes": expected_directory / "nodes.parquet",
            "links": expected_directory / "links.parquet",
            "turns": expected_directory / "turns.parquet",
        }
        assert paths == expected_paths
        assert all(path.is_file() for path in paths.values())
        assert set(expected_directory.iterdir()) == set(expected_paths.values())
