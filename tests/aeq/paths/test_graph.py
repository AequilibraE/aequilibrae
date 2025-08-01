import zipfile

import numpy as np
import pandas as pd
import pytest

from aequilibrae.paths import Graph
from aequilibrae.paths.results import PathResults
from aequilibrae.transit import Transit


@pytest.fixture(scope="function")
def test_graph(test_data_path):
    return test_data_path / "test_graph.aeg"


def graph_for_project(project):
    project.network.build_graphs(modes=["c"])
    return project.network.graphs["c"]


def test_prepare_graph(sioux_falls_example):
    graph = graph_for_project(sioux_falls_example)
    graph.prepare_graph(np.arange(5) + 1)


def test_prepare_graph_no_centroids(sioux_falls_example):
    graph = graph_for_project(sioux_falls_example)
    graph.prepare_graph()
    graph.set_graph("distance")
    graph.set_skimming("distance")


def test_set_graph(sioux_falls_example):
    graph = graph_for_project(sioux_falls_example)
    graph.set_graph(cost_field="distance")
    graph.set_blocked_centroid_flows(block_centroid_flows=True)
    assert graph.num_zones == 24, "Number of centroids not properly set"
    assert graph.num_links == 76, "Number of links not properly set"
    assert graph.num_nodes == 24, f"Number of nodes not properly set - {graph.num_nodes}"


def test_save_to_disk(sioux_falls_example):
    graph = graph_for_project(sioux_falls_example)
    graph.save_to_disk(sioux_falls_example.project_base_path / "aequilibrae_test_graph.aeg")


def test_load_from_disk(sioux_falls_example, test_graph):
    graph_file = sioux_falls_example.project_base_path / "aequilibrae_test_graph.aeg"
    graph = graph_for_project(sioux_falls_example)
    graph.save_to_disk(graph_file)
    reference_graph = Graph()
    reference_graph.load_from_disk(test_graph)

    new_graph = Graph()
    new_graph.load_from_disk(graph_file)


def test_available_skims(sioux_falls_example):
    graph = graph_for_project(sioux_falls_example)
    graph.prepare_graph(np.arange(5) + 1)
    avail = graph.available_skims()
    data_fields = [
        "distance",
        "name",
        "lanes",
        "capacity",
        "speed",
        "b",
        "free_flow_time",
        "power",
        "modes",
    ]
    assert all(i in avail for i in data_fields), "Skim availability with problems"


def test_compute_path(sioux_falls_example):
    graph = graph_for_project(sioux_falls_example)
    graph.prepare_graph()
    graph.set_graph("distance")
    graph.set_blocked_centroid_flows(False)

    res = graph.compute_path(1, 6)
    assert list(res.path) == [1, 4], "Number of path links is not correct"
    assert list(res.path_nodes) == [1, 2, 6], "Number of path nodes is not correct"


def test_compute_skims(sioux_falls_example):
    graph = graph_for_project(sioux_falls_example)
    graph.prepare_graph()
    graph.set_graph("distance")
    graph.set_skimming(["distance", "free_flow_time"])
    graph.set_blocked_centroid_flows(False)

    skm = graph.compute_skims()
    skims = skm.results.skims
    assert skims.cores == 2, "Number of cores is not correct"
    assert skims.names == ["distance", "free_flow_time"], "Matrices names are not correct"


def test_exclude_links(sioux_falls_example):
    # excludes a link before any setting or preparation
    graph = graph_for_project(sioux_falls_example)
    graph.set_blocked_centroid_flows(False)

    graph.set_graph("distance")
    r1 = PathResults()
    r1.prepare(graph)
    r1.compute_path(20, 21)
    assert list(r1.path) == [62]

    r1 = PathResults()
    graph.exclude_links([62])
    r1.prepare(graph)
    r1.compute_path(20, 21)
    assert list(r1.path) == [63, 69]


@pytest.fixture(scope="function")
def transit_data(coquimbo_example):
    return Transit(coquimbo_example)


@pytest.fixture(scope="function")
def transit_graph(transit_data):
    graph = transit_data.create_graph(
        with_outer_stop_transfers=False,
        with_walking_edges=False,
        blocking_centroid_flows=False,
        connector_method="nearest_neighbour",
    )
    return graph


def test_transit_graph_config(transit_graph):
    transit_graph_obj = transit_graph.to_transit_graph()
    assert transit_graph.config == transit_graph_obj._config


def test_transit_graph_od_node_mapping(transit_graph):
    transit_graph_obj = transit_graph.to_transit_graph()
    pd.testing.assert_frame_equal(transit_graph.od_node_mapping, transit_graph_obj.od_node_mapping)


@pytest.fixture(scope="function")
def compressed_graph(test_data_path, tmp_path):
    zipfile.ZipFile(test_data_path / "KaiTang.zip").extractall(tmp_path)

    link_df = pd.read_csv(tmp_path / "links_modified.csv")
    centroids_array = np.array([7, 8, 11])

    graph = Graph()
    graph.network = link_df
    graph.mode = "a"
    graph.prepare_graph(centroids_array)
    graph.set_blocked_centroid_flows(False)
    graph.set_graph("fft")
    return graph


def test_compressed_graph(compressed_graph):
    # Check the compressed links, links 4 and 5 should be collapsed into 2 links from 3 - 10 and 10 - 3.
    compressed_links = compressed_graph.graph[
        compressed_graph.graph.__compressed_id__.duplicated(keep=False)
        & (compressed_graph.graph.__compressed_id__ != compressed_graph.compact_graph.id.max() + 1)
    ]

    assert compressed_links.link_id.unique().tolist() == [4, 5]

    # Confirm these compacted links map back up to a contraction between the correct nodes
    assert compressed_graph.compact_all_nodes[
        compressed_graph.compact_graph[
            compressed_graph.compact_graph.id.isin(compressed_links.__compressed_id__.unique())
        ][["a_node", "b_node"]].values
    ].tolist() == [[3, 10], [10, 3]]


def test_dead_end_removal(compressed_graph):
    # The dead end remove should be able to remove links [30, 38]. In it's current state it is not able to remove
    # link 40 as it's a single direction link with no outgoing edges so its not possible to find the incoming edges
    # (in general) without a transposed graph representation.
    assert set(compressed_graph.dead_end_links) == set(
        compressed_graph.graph[compressed_graph.graph.dead_end == 1].link_id
    ) - {40}, "Dead end removal removed incorrect links"
