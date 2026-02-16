import os
from warnings import warn

import pytest
from shapely.geometry import box, Polygon

from aequilibrae.project.project import Project


def test_create_from_osm(empty_project):
    if os.environ.get("GITHUB_WORKFLOW", "ERROR") == "Code coverage":
        pytest.skip("Skipped check to not load OSM servers")

    empty_project.network.create_from_osm(model_area=box(-112.185, 36.59, -112.179, 36.60))

    with empty_project.db_connection as conn:
        lks = conn.execute("""select count(*) from links""").fetchone()[0]
        osmids = conn.execute("""select count(distinct osm_id) from links""").fetchone()[0]

        if osmids == 0:
            warn("COULD NOT RETRIEVE DATA FROM OSM")
            return

        if osmids >= lks:
            pytest.fail("OSM links not broken down properly")

        nds = conn.execute("""select count(*) from nodes""").fetchone()[0]

        if lks > nds:
            pytest.fail("We imported more links than nodes. Something wrong here")


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
