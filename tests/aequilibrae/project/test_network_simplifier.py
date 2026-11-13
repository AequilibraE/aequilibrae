import pytest
from shapely.geometry import LineString

from aequilibrae.project.tools.network_simplifier import NetworkSimplifier


@pytest.fixture(scope="function")
def project_with_graph(nauru_example):
    remaining_links = [899, 900, 901, 902, 903, 1042, 1043, 1159, 1160]
    remaining_links += list(range(171, 222))

    with nauru_example.db_connection as conn:
        qry = f"DELETE FROM links WHERE link_id NOT IN {tuple(remaining_links)};"
        conn.execute(qry)
        conn.commit()

        # Let's create a centroid to build a graph
        arbitrary_node = conn.execute("select node_id from nodes limit 1").fetchone()[0]
        nodes = nauru_example.network.nodes
        nodes.update(arbitrary_node, is_centroid=1)

    mode = "c"

    network = nauru_example.network.build()
    graph = network.build_graph(mode, "distance", skimming=["distance"], compact=True, block_centroid_flows=False)

    yield graph


def test_simplify(project_with_graph):
    net = NetworkSimplifier()
    graph = project_with_graph

    links_before = net.link_layer.shape[0]
    nodes_before = net.network.nodes.data.shape[0]

    net.simplify(graph)

    assert net.graph is graph
    assert links_before > net.network.links.data.shape[0]
    assert nodes_before > net.network.nodes.data.shape[0]


def test_simplify_uses_ordered_network_expansions(empty_project):
    # Link IDs differ from traversal order; the first geometry opposes the chain.
    with empty_project.db_connection as conn:
        conn.execute("ALTER TABLE links ADD COLUMN lanes_ab INTEGER")
        conn.execute("ALTER TABLE links ADD COLUMN lanes_ba INTEGER")
        conn.executemany(
            "INSERT INTO links (link_id, direction, modes, link_type, speed_ab, speed_ba, "
            "lanes_ab, lanes_ba, capacity_ab, capacity_ba, geometry) "
            "VALUES (?, ?, 'c', 'default', ?, ?, ?, ?, ?, ?, GeomFromWKB(?, 4326))",
            [
                (30, -1, 10.0, 40.0, 1, 2, 50.0, 100.0, LineString([(0.02, 0), (0, 0)]).wkb),
                (10, 1, 42.0, 10.5, 2, 1, 300.0, 90.0, LineString([(0.02, 0), (0.03, 0)]).wkb),
            ],
        )
        conn.execute("UPDATE nodes SET is_centroid=1 WHERE node_id IN (SELECT b_node FROM links)")
    empty_project.network.refresh_schemas()

    graph = empty_project.network.build().build_graph("c", "distance", compact=True)
    NetworkSimplifier(empty_project).simplify(graph)

    links = empty_project.network.links.data
    assert len(links) == 1
    merged = links.iloc[0]
    assert list(merged.geometry.coords) == [(0, 0), (0.02, 0), (0.03, 0)]
    assert merged.direction == 1
    assert merged.speed_ab == pytest.approx((40 * 2 + 42) / 3)
    assert merged.speed_ba == pytest.approx((10 * 2 + 10.5) / 3)
    assert merged.capacity_ab == pytest.approx((100 * 2 + 300) / 3)
    assert merged.capacity_ba == pytest.approx((50 * 2 + 90) / 3)


def test_collapse_links_into_nodes(project_with_graph):
    net = NetworkSimplifier()

    links_before = net.link_layer.shape[0]
    nodes_before = net.network.nodes.data.shape[0]

    net.collapse_links_into_nodes([903])

    assert links_before > net.link_layer.shape[0]
    assert nodes_before > net.network.nodes.data.shape[0]
