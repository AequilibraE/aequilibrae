import os
import tempfile
import uuid
import pandas as pd

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

    graph.block_centroid_flows = False
    graph.exclude_links([5, 12, 19])  # Disconnect nodes 1, 2, and 6 from the reset of the graph
    conn_test = ConnectivityAnalysis(graph)
    conn_test.execute()

    # Construct the disconnected pairs. 1, 2, and 6 are not reachable from anywhere in
    # the graph other than them selves, but they can reach everywhere else.
    origins = []
    destinations = []
    disconnected = (1, 2, 6)
    for o in graph.centroids:
        if o not in disconnected:
            origins.extend((o,) * len(disconnected))
            destinations.extend(disconnected)
    df = pd.DataFrame({"origin": origins, "destination": destinations}).astype("int64")

    results = conn_test.disconnected_pairs.reset_index(drop=True)
    pd.testing.assert_frame_equal(results, df, check_dtype=False, check_index_type=False)

    project.close()
