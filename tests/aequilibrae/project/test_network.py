import pytest
from warnings import warn

from shapely.geometry import Polygon
from shapely.geometry import Polygon, box


def test_create_from_osm_was_removed(empty_project):
    """``create_from_osm`` was removed; calling it raises with a migration hint."""
    with pytest.raises(AttributeError, match="import_from_osm"):
        empty_project.network.create_from_osm(place_name="Nauru")


def test_import_from_osm_via_pbf(empty_project):
    pytest.importorskip("pyrosm")
    from pyrosm import get_data

    empty_project.network.import_from_osm(
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
    node = nodes.get(1)
    node.is_centroid = 0
    node.save()

    items = sioux_falls_test.network.count_centroids()
    assert items == 23, "Wrong number of centroids found"


def test_count_links(sioux_falls_test):
    items = sioux_falls_test.network.count_links()
    assert items == 76, "Wrong number of links found"


def test_count_nodes(sioux_falls_test):
    items = sioux_falls_test.network.count_nodes()
    assert items == 24, "Wrong number of nodes found"


def test_build_graphs_with_polygons(sioux_falls_test):
    coords = ((-96.75, 43.50), (-96.75, 43.55), (-96.70, 43.55), (-96.70, 43.50), (-96.75, 43.50))
    polygon = Polygon(coords)

    fields = ["distance"]
    modes = ["c"]

    sioux_falls_test.network.build_graphs(fields, modes, polygon)
    assert len(sioux_falls_test.network.graphs) == 1

    g = sioux_falls_test.network.graphs["c"]
    assert g.num_nodes == 19
    assert g.num_links == 52

    existing_nodes = [i for i in range(1, 25) if i not in [1, 2, 3, 6, 7]]
    assert list(g.centroids) == existing_nodes


def test_build_graphs_without_polygons(sioux_falls_test):
    sioux_falls_test.network.build_graphs()
    assert len(sioux_falls_test.network.graphs) == 3

    g = sioux_falls_test.network.graphs["c"]
    assert g.num_nodes == 24
    assert g.num_links == 76
    assert list(g.centroids) == list(range(1, 25))
