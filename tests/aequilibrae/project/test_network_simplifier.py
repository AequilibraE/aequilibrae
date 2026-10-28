import pytest

from aequilibrae.paths.graph import GraphBase, _prepare_network_graph
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

    network = nauru_example.network
    graph = network.build_graph(mode, "distance", skimming=["distance"], compact=True, block_centroid_flows=False)

    yield graph


@pytest.mark.parametrize("legacy", [False, True])
def test_simplify(project_with_graph, monkeypatch, legacy):
    net = NetworkSimplifier()
    graph = project_with_graph
    if legacy:
        links, centroids, turns, allow_uturns = net.network._load_graph_data()
        graph = GraphBase()
        graph.network = links
        _prepare_network_graph(graph, "c", centroids, turns, False, allow_uturns)
        graph.set_graph("distance")

    links_before = net.link_layer.shape[0]
    nodes_before = net.network.nodes.data.shape[0]

    def unexpected_build(*args, **kwargs):
        raise AssertionError("simplify must use the supplied graph")

    monkeypatch.setattr(net.network, "build_graph", unexpected_build)
    net.simplify(graph)

    assert net.graph is graph
    assert links_before > net.network.links.data.shape[0]
    assert nodes_before > net.network.nodes.data.shape[0]


def test_simplify_rejects_full_prepared_graph(project_with_graph):
    net = NetworkSimplifier()
    graph = net.network.build_graph("c", "distance", compact=False)
    links_before = net.link_layer.shape[0]

    with pytest.raises(ValueError, match="requires a compact graph"):
        net.simplify(graph)

    assert net.network.links.data.shape[0] == links_before


def test_collapse_links_into_nodes(project_with_graph):
    net = NetworkSimplifier()

    links_before = net.link_layer.shape[0]
    nodes_before = net.network.nodes.data.shape[0]

    net.collapse_links_into_nodes([903])

    assert links_before > net.link_layer.shape[0]
    assert nodes_before > net.network.nodes.data.shape[0]
