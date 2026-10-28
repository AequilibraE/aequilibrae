import numpy as np
import pytest
from shapely.geometry import Polygon


def test_import_from_osm_via_pbf(empty_project):
    pytest.importorskip("pyrosm")
    from pyrosm import get_data

    empty_project.network.importer.osm(
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
    items = sioux_falls_test.network.count_centroids()
    assert items == 24, "Wrong number of centroids found"

    nodes = sioux_falls_test.network.nodes
    nodes.update(1, is_centroid=0)

    items = sioux_falls_test.network.count_centroids()
    assert items == 23, "Wrong number of centroids found"


def test_count_links(sioux_falls_test):
    items = sioux_falls_test.network.count_links()
    assert items == 76, "Wrong number of links found"


def test_count_nodes(sioux_falls_test):
    items = sioux_falls_test.network.count_nodes()
    assert items == 24, "Wrong number of nodes found"


def test_build_with_polygons(sioux_falls_test):
    coords = ((-96.75, 43.50), (-96.75, 43.55), (-96.70, 43.55), (-96.70, 43.50), (-96.75, 43.50))
    polygon = Polygon(coords)

    fields = ["distance"]

    network = sioux_falls_test.network

    g = network.build_graph(mode="c", cost="distance", fields=fields, limit_to_area=polygon)
    assert g.num_nodes == 19
    assert g.num_links == 52

    existing_nodes = [i for i in range(1, 25) if i not in [1, 2, 3, 6, 7]]
    assert list(g.centroids) == existing_nodes


def test_build_without_polygons(sioux_falls_test):
    network = sioux_falls_test.network
    assert len(network.list_modes()) == 3

    g = network.build_graph(mode="c", cost="distance")
    assert g.num_nodes == 24
    assert g.num_links == 76
    assert list(g.centroids) == list(range(1, 25))


def test_build_applies_project_uturn_setting_without_turns(sioux_falls_example):
    assert len(sioux_falls_example.network.turn_restrictions) == 0
    sioux_falls_example.about.update(infoname="allow_uturns", infovalue=1)

    graph = sioux_falls_example.network.build_graph("c", "distance")
    assert graph.allow_path_uturns
    assert not graph.has_turn_restrictions


def test_build_filters_turns_by_mode(sioux_falls_example):
    network = sioux_falls_example.network
    kept = network.turn_restrictions.insert(from_node=1, via_node=2, to_node=6, penalty=5.0, modes="c")
    network.turn_restrictions.insert(from_node=1, via_node=2, to_node=6, penalty=9.0, modes="w")

    source = network.build(modes=["c"])
    graph = source.build_graph("c", "distance", block_centroid_flows=False)

    assert source.turn_restrictions.restriction_id.tolist() == [kept]
    assert graph.full_turn_penalties[np.isfinite(graph.full_turn_penalties)].tolist() == [5.0]


def test_build_filters_turns_by_area(sioux_falls_example):
    network = sioux_falls_example.network
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
    incoming = (graph.full_node_ids[graph.full_tails] == 4) & (graph.full_node_ids[graph.full_heads] == 5)
    assert not incoming.any()


def test_build_rejects_nonexistent_directed_turn_leg(sioux_falls_example):
    network = sioux_falls_example.network
    network.turn_restrictions.insert(from_node=1, via_node=4, to_node=5, penalty=5.0, modes="c")

    with pytest.raises(ValueError, match="missing directed legs from_node -> via_node"):
        network.build_graph("c", "distance")
