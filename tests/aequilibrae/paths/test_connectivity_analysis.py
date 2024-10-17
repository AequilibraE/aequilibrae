import os
import tempfile
import uuid

from aequilibrae.paths.connectivity_analysis import ConnectivityAnalysis
from aequilibrae.utils.create_example import create_example


def test_connectivity_analysis():
    project = create_example(os.path.join(tempfile.gettempdir(), uuid.uuid4().hex))
    network = project.network
    network.build_graphs()
    graph = network.graphs["c"]
    graph.set_graph(cost_field="distance")
    graph.block_centroid_flows = False
    conn_test = ConnectivityAnalysis(graph)
    conn_test.execute()

    assert conn_test.disconnected_pairs.shape[0] == 0

    graph.block_centroid_flows = True
    conn_test = ConnectivityAnalysis(graph)
    conn_test.execute()

    # We can only reach the nodes each node is directly connected to
    assert conn_test.disconnected_pairs.shape[0] == graph.num_zones * (graph.num_zones - 1) - graph.network.shape[0]

    project.close()
