import pytest

from aequilibrae.utils.create_example import create_example
from aequilibrae.project.tools.network_simplifier import NetworkSimplifier


@pytest.fixture
def project_with_graph(create_path):
    project = create_example(create_path, "nauru")

    mode = "c"

    # Let's create a centroid to build a graph
    centroid_count = project.conn.execute("select count(*) from nodes where is_centroid=1").fetchone()[0]

    if centroid_count == 0:
        arbitrary_node = project.conn.execute("select node_id from nodes limit 1").fetchone()[0]
        nodes = project.network.nodes
        nd = nodes.get(arbitrary_node)
        nd.is_centroid = 1
        nd.save()

    network = project.network
    network.build_graphs(modes=[mode])
    graph = network.graphs[mode]
    graph.set_graph("distance")
    graph.set_skimming("distance")
    graph.set_blocked_centroid_flows(False)

    yield graph
    project.close()


def test_simplify(project_with_graph):
    net = NetworkSimplifier()

    net.simplify(project_with_graph)
    net.rebuild_network()


def test_collapse_links_into_nodes():
    net = NetworkSimplifier()
    net.collapse_links_into_nodes([])
