import numpy as np
import pytest
from shapely.geometry import Polygon

from aequilibrae.matrix import from_dict
from aequilibrae.traffic import TrafficAssignment, TrafficClass
from aequilibrae.traffic.vdf import bpr


def test_import_from_osm_via_pbf(empty_project):
    pytest.importorskip("pyrosm")
    from pyrosm import get_data

    empty_project.tables.importer.osm(
        pbf_path=get_data("test_pbf"),
        modes=("car",),
        simplify=False,
    )
    with empty_project.db_connection as conn:
        n_links = conn.execute("SELECT count(*) FROM links").fetchone()[0]
        n_nodes = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
    assert n_links > 10
    assert n_nodes > 10


def test_count_centroids(sioux_falls_test):
    items = sioux_falls_test.tables.count_centroids()
    assert items == 24, "Wrong number of centroids found"

    nodes = sioux_falls_test.tables.nodes
    nodes.update(1, is_centroid=0)

    items = sioux_falls_test.tables.count_centroids()
    assert items == 23, "Wrong number of centroids found"


def test_count_links(sioux_falls_test):
    items = sioux_falls_test.tables.count_links()
    assert items == 76, "Wrong number of links found"


def test_count_nodes(sioux_falls_test):
    items = sioux_falls_test.tables.count_nodes()
    assert items == 24, "Wrong number of nodes found"


def test_build_with_polygons(sioux_falls_test):
    coords = ((-96.75, 43.50), (-96.75, 43.55), (-96.70, 43.55), (-96.70, 43.50), (-96.75, 43.50))
    polygon = Polygon(coords)

    fields = ["distance"]

    network = sioux_falls_test.tables

    g = network.build_graph(mode="c", cost="distance", fields=fields, limit_to_area=polygon)
    assert g.num_nodes == 19
    assert g.num_links == 52

    existing_nodes = [i for i in range(1, 25) if i not in [1, 2, 3, 6, 7]]
    assert list(g.centroids) == existing_nodes


def test_build_without_polygons(sioux_falls_test):
    network = sioux_falls_test.tables
    assert len(network.list_modes()) == 3

    g = network.build_graph(mode="c", cost="distance")
    assert g.num_nodes == 24
    assert g.num_links == 76
    assert list(g.centroids) == list(range(1, 25))


def test_build_applies_project_uturn_setting_without_turns(sioux_falls_example):
    assert len(sioux_falls_example.tables.turn_restrictions) == 0
    sioux_falls_example.about.update(infoname="allow_uturns", infovalue=1)

    network = sioux_falls_example.tables.build()
    assert network.allow_uturns
    graph = network.build_graph("c", "distance")
    assert not graph.has_turn_restrictions


def test_build_filters_turns_by_mode(sioux_falls_example):
    network = sioux_falls_example.tables
    kept = network.turn_restrictions.insert(from_node=1, via_node=2, to_node=6, penalty=5.0, modes="c")
    network.turn_restrictions.insert(from_node=1, via_node=2, to_node=6, penalty=9.0, modes="w")

    source = network.build(modes=["c"])
    graph = source.build_graph("c", "distance", block_centroid_flows=False)

    assert source.turn_restrictions.restriction_id.tolist() == [kept]
    assert graph.routing.turn_penalties[np.isfinite(graph.routing.turn_penalties)].tolist() == [5.0]


def test_build_filters_turns_by_area(sioux_falls_example):
    network = sioux_falls_example.tables
    kept = network.turn_restrictions.insert(from_node=5, via_node=9, to_node=10, penalty=5.0, modes="c")
    network.turn_restrictions.insert(from_node=1, via_node=2, to_node=6, penalty=9.0, modes="c")
    network.turn_restrictions.insert(from_node=4, via_node=5, to_node=9, penalty=9.0, modes="c")
    polygon = Polygon(((-96.75, 43.50), (-96.75, 43.55), (-96.70, 43.55), (-96.70, 43.50)))

    source = network.build(modes=["c"], limit_to_area=polygon)
    graph = source.build_graph("c", "distance")
    assert source.turn_restrictions.restriction_id.tolist() == [kept]
    assert 1 not in graph.all_nodes
    # All three nodes remain, but the incoming leg 4 -> 5 is outside the selection.
    assert {4, 5, 9}.issubset(graph.all_nodes)
    incoming = (graph.all_nodes[graph.routing.tails] == 4) & (graph.all_nodes[graph.routing.heads] == 5)
    assert not incoming.any()


@pytest.fixture
def network_with_mode_excluded_centroid(sioux_falls_example):
    network = sioux_falls_example.tables
    other_mode = next(mode for mode in network.list_modes() if mode != "c")
    with sioux_falls_example.db_connection as conn:
        conn.execute("UPDATE links SET modes = ? WHERE a_node = ? OR b_node = ?", (other_mode, 1, 1))

    return network


def test_build_preserves_centroids_across_modes(network_with_mode_excluded_centroid):
    network = network_with_mode_excluded_centroid
    source = network.build(modes=["c"])

    assert source.centroids.tolist() == list(range(1, 25))
    assert source.build_graph("c", "distance").centroids.tolist() == list(range(1, 25))
    source = network.build()
    for mode in network.list_modes():
        assert source.build_graph(mode, "distance").centroids.tolist() == list(range(1, 25))


def test_build_graph_remove_dead_ends_option(sioux_falls_example):
    network = sioux_falls_example.tables
    empty_mode = next(mode for mode in network.list_modes() if mode != "c")
    with sioux_falls_example.db_connection as conn:
        conn.execute("UPDATE links SET modes = ?", ("c",))

    graph = network.build_graph(empty_mode, "distance", compact=False, remove_dead_ends=True)

    assert graph.centroids.tolist() == list(range(1, 25))
    assert graph.num_links == 0


def test_build_preserves_explicit_mode_excluded_centroid(network_with_mode_excluded_centroid):
    graph = network_with_mode_excluded_centroid.build_graph("c", "distance", centroids=np.array([1, 2]))
    assert graph.centroids.tolist() == [1, 2]


def test_assignment_reports_demand_at_mode_excluded_centroid_as_unassigned(network_with_mode_excluded_centroid):
    source = network_with_mode_excluded_centroid.build(modes=["c"])
    values = np.zeros((24, 24))
    values[0, 1], values[1, 0] = 5.0, 3.0
    demand = from_dict({"flow": values}, source.centroids)
    traffic = TrafficClass(mode="c", cost="free_flow_time", demand=demand, skimming=("free_flow_time",))
    assignment = TrafficAssignment(
        network=source,
        classes=[traffic],
        time_field="free_flow_time",
        capacity_field="capacity",
        vdf=bpr,
        algorithm="all-or-nothing",
    )
    assignment.execute(cores=1, elementwise_cores=1)
    results = assignment.class_results["c"]

    assert results.unassigned_demand == 8.0
    assert not np.any(results.link_loads)
    assert np.isinf(results.skims.matrices["free_flow_time"][0, 1])
    assert np.isinf(results.skims.matrices["free_flow_time"][1, 0])


@pytest.mark.parametrize(
    "fields",
    [
        None,
        [],
        ["review_cost"],
        ["review_cost_ab", "review_cost_ba"],
        ["review_cost", "review_cost_ab", "review_cost_ba"],
    ],
)
def test_field_selection_preserves_directional_cost_precedence(sioux_falls_example, fields):
    project = sioux_falls_example
    with project.db_connection as conn:
        conn.execute("ALTER TABLE links ADD COLUMN review_cost REAL")
        conn.execute("ALTER TABLE links ADD COLUMN review_cost_ab REAL")
        conn.execute("ALTER TABLE links ADD COLUMN review_cost_ba REAL")
        conn.execute("UPDATE links SET review_cost = ?, review_cost_ab = ?, review_cost_ba = ?", (1.0, 2.0, 3.0))
        conn.execute("UPDATE links SET direction = ? WHERE link_id = ?", (0, 1))

    graph = project.tables.build_graph("c", "review_cost", fields=fields)
    network_costs = np.where(graph.mapping.network.directions == 1, 2.0, 3.0)
    np.testing.assert_array_equal(graph.cost, graph.mapping.aggregate_costs(network_costs))

    source = project.tables.build(fields=["review_cost"])
    np.testing.assert_array_equal(source.field_values("review_cost"), np.where(source.directions == 1, 2.0, 3.0))


@pytest.mark.parametrize("directional_column", [None, "review_cost_ab", "review_cost_ba"])
def test_selected_field_falls_back_to_base_per_direction(sioux_falls_example, directional_column):
    project = sioux_falls_example
    with project.db_connection as conn:
        conn.execute("ALTER TABLE links ADD COLUMN review_cost REAL")
        conn.execute("UPDATE links SET review_cost = ?", (1.0,))
        conn.execute("UPDATE links SET direction = ? WHERE link_id = ?", (0, 1))
        if directional_column == "review_cost_ab":
            conn.execute("ALTER TABLE links ADD COLUMN review_cost_ab REAL")
            conn.execute("UPDATE links SET review_cost_ab = ?", (2.0,))
        elif directional_column == "review_cost_ba":
            conn.execute("ALTER TABLE links ADD COLUMN review_cost_ba REAL")
            conn.execute("UPDATE links SET review_cost_ba = ?", (3.0,))

    full = project.tables.build()
    selected = project.tables.build(fields=["review_cost"])
    expected = np.ones(full.mapping.link_count)
    if directional_column is not None:
        direction, value = (1, 2.0) if directional_column.endswith("_ab") else (-1, 3.0)
        expected[full.directions == direction] = value
    np.testing.assert_array_equal(full.field_values("review_cost"), expected)
    np.testing.assert_array_equal(selected.field_values("review_cost"), expected)


def test_build_rejects_nonexistent_directed_turn_leg(sioux_falls_example):
    network = sioux_falls_example.tables
    network.turn_restrictions.insert(from_node=1, via_node=4, to_node=5, penalty=5.0, modes="c")

    with pytest.raises(ValueError, match="missing directed legs from_node -> via_node"):
        network.build_graph("c", "distance")
