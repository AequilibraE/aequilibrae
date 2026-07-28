import zipfile
from collections import defaultdict

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


def sample_turn_from_path(graph, origin, destination):
    res = graph.compute_path(origin, destination)
    path_nodes = [int(x) for x in res.path_nodes]
    assert len(path_nodes) >= 3
    return path_nodes[0], path_nodes[1], path_nodes[2]


def test_upper_case_variables(sioux_falls_example):
    graph = graph_for_project(sioux_falls_example)
    network = graph.network
    network.columns = network.columns.str.upper()
    g = Graph()
    g.network = network
    assert g.network.columns.tolist() == graph.network.columns.tolist(), "Graph columns are not lower case"

    g.prepare_graph()
    g.set_graph("DiStAnce")
    assert g.cost_field == "distance", "Graph cost field is not set to distance lower case"

    g.set_skimming("DiStAnce")
    assert g.skim_fields == ["distance"], "Graph skim fields are not set to distance lower case"


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


def test_set_turn_restrictions_rejects_negative_penalty(sioux_falls_example):
    graph = graph_for_project(sioux_falls_example)
    graph.prepare_graph()
    graph.set_graph("distance")
    graph.set_blocked_centroid_flows(False)

    from_node, via_node, to_node = sample_turn_from_path(graph, 1, 6)
    turn_restrictions = pd.DataFrame(
        {"from_node": [from_node], "via_node": [via_node], "to_node": [to_node], "penalty": [-1.0]}
    )

    with pytest.raises(ValueError, match="Negative turn penalties"):
        graph.set_turn_restrictions(turn_restrictions)


def test_a_star_raises_with_turn_restrictions(sioux_falls_example):
    graph = graph_for_project(sioux_falls_example)
    graph.prepare_graph()
    graph.set_graph("distance")
    graph.set_blocked_centroid_flows(False)

    from_node, via_node, to_node = sample_turn_from_path(graph, 1, 6)
    turn_restrictions = pd.DataFrame(
        {"from_node": [from_node], "via_node": [via_node], "to_node": [to_node], "penalty": [np.inf]}
    )
    graph.set_turn_restrictions(turn_restrictions)

    with pytest.raises(RuntimeError, match="not compatible"):
        graph.compute_path(1, 6, a_star=True)


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
    origin = 20
    destination = 21
    graph.set_graph("distance")
    r1 = PathResults(graph, origin, destination)
    assert list(r1.path) == [62]

    graph.exclude_links([62])
    r1 = PathResults(graph, origin, destination)
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


def test_turn_restrictions_match_networkx(coquimbo_example):
    nx = pytest.importorskip("networkx")

    coquimbo_example.network.build_graphs(modes=["c"])
    graph = coquimbo_example.network.graphs["c"]
    graph.prepare_graph()
    graph.set_graph("distance")
    graph.set_blocked_centroid_flows(False)

    origin = 32343
    destination = 22041

    graph_df = graph.graph[["id", "a_node", "b_node", "link_id", "direction", "distance"]].copy()
    graph_df["a_node"] = graph.all_nodes[graph_df["a_node"].to_numpy(dtype=np.int64)]
    graph_df["b_node"] = graph.all_nodes[graph_df["b_node"].to_numpy(dtype=np.int64)]

    arc_lookup = {}
    for row in graph_df.itertuples(index=False):
        key = (int(row.a_node), int(row.b_node), int(row.link_id))
        arc_lookup.setdefault(key, (int(row.id), int(row.direction), float(row.distance)))

    def aeq_directed_path(path_nodes, path_links):
        directed = []
        total = 0.0
        for a_node, b_node, link_id in zip(path_nodes[:-1], path_nodes[1:], path_links, strict=False):
            key = (int(a_node), int(b_node), int(link_id))
            arc_id, direction, distance = arc_lookup[key]
            directed.append((arc_id, int(link_id), direction))
            total += distance
        return directed, total

    base_graph = nx.MultiDiGraph()
    for row in graph_df.itertuples(index=False):
        base_graph.add_edge(int(row.a_node), int(row.b_node), key=int(row.id), weight=float(row.distance))

    base_path_nx = nx.shortest_path(base_graph, origin, destination, weight="weight")
    assert base_path_nx is not None
    base_cost_nx = nx.shortest_path_length(base_graph, origin, destination, weight="weight")

    res_before = graph.compute_path(origin, destination)
    _, base_cost_aeq = aeq_directed_path(res_before.path_nodes, res_before.path)
    assert base_cost_aeq == pytest.approx(base_cost_nx)

    path_nodes = [int(x) for x in res_before.path_nodes]
    triples = [(path_nodes[i], path_nodes[i + 1], path_nodes[i + 2]) for i in range(max(0, len(path_nodes) - 2))]
    assert len(triples) >= 4
    turn_restrictions = pd.DataFrame(
        {
            "from_node": [t[0] for t in triples[:4]],
            "via_node": [t[1] for t in triples[:4]],
            "to_node": [t[2] for t in triples[:4]],
            "penalty": [np.nan, np.nan, np.nan, np.nan],
        }
    )

    graph.set_turn_restrictions(turn_restrictions, allow_path_uturns=False)
    res_after = graph.compute_path(origin, destination)
    directed_after_aeq, restricted_cost_aeq = aeq_directed_path(res_after.path_nodes, res_after.path)

    penalty_lookup = {
        (int(r.from_node), int(r.via_node), int(r.to_node)): r.penalty
        for r in turn_restrictions.itertuples(index=False)
    }
    prohibited = {key for key, penalty in penalty_lookup.items() if pd.isna(penalty) or np.isposinf(float(penalty))}

    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    arc_meta = {}
    for row in graph_df.itertuples(index=False):
        arc_id = int(row.id)
        a_node = int(row.a_node)
        b_node = int(row.b_node)
        arc_meta[arc_id] = {
            "a_node": a_node,
            "b_node": b_node,
            "link_id": int(row.link_id),
            "direction": int(row.direction),
            "distance": float(row.distance),
        }
        outgoing[a_node].append(arc_id)
        incoming[b_node].append(arc_id)

    state_graph = nx.DiGraph()
    source_state = "__source__"
    sink_state = "__sink__"

    for arc_id in outgoing[origin]:
        state_graph.add_edge(source_state, arc_id, weight=arc_meta[arc_id]["distance"])

    for node in set(incoming.keys()) & set(outgoing.keys()):
        for in_arc in incoming[node]:
            in_meta = arc_meta[in_arc]
            for out_arc in outgoing[node]:
                out_meta = arc_meta[out_arc]
                if out_meta["b_node"] == in_meta["a_node"]:
                    continue
                transition_key = (in_meta["a_node"], in_meta["b_node"], out_meta["b_node"])
                if transition_key in prohibited:
                    continue
                penalty = penalty_lookup.get(transition_key, 0.0)
                if pd.isna(penalty) or np.isposinf(float(penalty)):
                    continue
                state_graph.add_edge(in_arc, out_arc, weight=out_meta["distance"] + float(penalty))

    for arc_id in incoming[destination]:
        state_graph.add_edge(arc_id, sink_state, weight=0.0)

    state_path = nx.shortest_path(state_graph, source_state, sink_state, weight="weight")
    assert state_path is not None
    restricted_cost_nx = nx.shortest_path_length(state_graph, source_state, sink_state, weight="weight")

    directed_after_triples = [
        (int(res_after.path_nodes[i]), int(res_after.path_nodes[i + 1]), int(res_after.path_nodes[i + 2]))
        for i in range(max(0, len(res_after.path_nodes) - 2))
    ]
    for from_node, via_node, to_node in prohibited:
        for triple in directed_after_triples:
            if triple == (from_node, via_node, to_node):
                raise AssertionError("AequilibraE path contains a prohibited turn")

    assert restricted_cost_aeq == pytest.approx(restricted_cost_nx)
