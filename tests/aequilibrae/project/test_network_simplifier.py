import pytest

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
        nd = nodes.get(arbitrary_node)
        nd.is_centroid = 1
        nd.save()

    mode = "c"

    network = nauru_example.network
    network.build_graphs(modes=[mode])
    graph = network.graphs[mode]
    graph.set_graph("distance")
    graph.set_skimming("distance")
    graph.set_blocked_centroid_flows(False)

    yield graph


def test_simplify(project_with_graph):
    net = NetworkSimplifier()

    links_before = net.link_layer.shape[0]
    nodes_before = net.network.nodes.data.shape[0]

    net.simplify(project_with_graph)
    net.rebuild_network()

    assert links_before > net.network.links.data.shape[0]
    assert nodes_before > net.network.nodes.data.shape[0]


def test_collapse_links_into_nodes(project_with_graph):
    net = NetworkSimplifier()

    links_before = net.link_layer.shape[0]
    nodes_before = net.network.nodes.data.shape[0]

    net.collapse_links_into_nodes([903])
    net.rebuild_network()

    assert links_before > net.link_layer.shape[0]
    assert nodes_before > net.network.nodes.data.shape[0]
