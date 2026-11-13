import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from aequilibrae.paths import Network, estimate_heuristic_scale
from aequilibrae.paths.results import PathResults
from aequilibrae.transit import Transit


def build_graph(links, centroids=None, **options):
    """Build a graph from source tables, preserving any isolated centroids."""
    return Network(pd.DataFrame(links), centroids=centroids).build_graph("c", **options)


def sample_turn_from_path(graph, origin, destination):
    res = graph.compute_path(origin, destination)
    path_nodes = [int(x) for x in res.path_nodes]
    assert len(path_nodes) >= 3
    return path_nodes[0], path_nodes[1], path_nodes[2]


def test_prepare_graph(sioux_falls_example):
    graph = sioux_falls_example.network.build_graph("c", "distance", centroids=np.arange(5) + 1)
    np.testing.assert_array_equal(graph.centroids, np.arange(5) + 1)


def test_prepare_graph_no_centroids(sioux_falls_example):
    network = Network(sioux_falls_example.network.build().links)
    graph = network.build_graph("c", "distance", skimming=["distance"])
    assert graph.num_zones == 0
    assert graph.skim_fields == ["distance"]


def test_prepare_graph_first_node_has_no_outgoing_links():
    graph = build_graph({"link_id": [1, 2], "a_node": [2, 2], "b_node": [1, 3], "direction": [1, 1]}, [1])
    first = graph.mapping.graph_node(1)
    assert first == 0
    assert graph.routing.fs[first] == graph.routing.fs[first + 1]
    np.testing.assert_array_equal(np.sort(graph.all_nodes), [1, 2, 3])


@pytest.mark.parametrize("centroids", [[99, 1, 3], [99, 98, 1, 3]])
def test_directed_graph_offsets_with_isolated_first_centroids(centroids):
    links = {"link_id": [1, 2], "a_node": [1, 2], "b_node": [2, 3], "direction": [1, 1]}
    # Test adjacency construction without invoking dead-end removal or compression.
    graph = build_graph(links, centroids)
    fs = graph.routing.fs
    tails = graph.mapping.network_node_to_graph[graph.mapping.network.link_tails[graph.mapping.expansion_links]]
    assert fs[0] == 0
    assert fs[-1] == graph.num_links
    np.testing.assert_array_equal(np.diff(fs), np.bincount(tails, minlength=graph.num_nodes))


@pytest.mark.parametrize("centroids", [[99, 1, 3], [1, 2, 3, 99]])
def test_compression_preserves_isolated_centroids(centroids):
    graph = build_graph(
        {"link_id": [1, 2], "a_node": [1, 2], "b_node": [2, 3], "direction": [1, 1], "cost": [1.0, 1.0]},
        centroids,
        cost="cost",
        skimming=["cost"],
        compact=True,
        block_centroid_flows=False,
    )
    np.testing.assert_array_equal(graph.centroids, centroids)
    np.testing.assert_array_equal(graph.mapping.graph_node(centroids), np.arange(len(centroids)))
    isolated = graph.mapping.graph_node(99)
    assert graph.routing.fs[isolated] == graph.routing.fs[isolated + 1]
    skims = graph.compute_skims().results.skims["cost"]
    assert skims[centroids.index(1), centroids.index(3)] == 2.0
    assert np.isinf(skims[centroids.index(1), centroids.index(99)])


@pytest.mark.parametrize(
    "links, centroids",
    [
        (
            {"link_id": [1], "a_node": [10], "b_node": [20], "direction": [1], "cost": [1.0]},
            [99, 100],
        ),
        (
            {"link_id": [1, 2], "a_node": [10, 20], "b_node": [20, 30], "direction": [1, 1], "cost": [1.0, 1.0]},
            [99, 100],
        ),
        (
            {"link_id": [1, 2], "a_node": [1, 2], "b_node": [3, 3], "direction": [1, 1], "cost": [1.0, 1.0]},
            [1, 2],
        ),
    ],
)
def test_empty_compact_graph(links, centroids):
    graph = build_graph(links, centroids, cost="cost", skimming=["cost"], compact=True, remove_dead_ends=True)
    assert graph.num_links == 0

    np.testing.assert_array_equal(graph.all_nodes, centroids)
    np.testing.assert_array_equal(graph.mapping.graph_node(centroids), [0, 1])
    np.testing.assert_array_equal(graph.routing.fs, [0, 0, 0])
    np.testing.assert_array_equal(graph.mapping.network_link_to_graph, -1)
    np.testing.assert_array_equal(graph.mapping.expansion_offsets, [0])
    assert graph.mapping.expansion_links.size == 0
    assert np.isinf(graph.compute_skims().results.skims["cost"][0, 1])


def test_compaction_preserves_cycle_needed_to_avoid_turn_ban():
    links = {
        "link_id": [1, 2, 3, 4, 5],
        "a_node": [1, 2, 2, 4, 5],
        "b_node": [2, 6, 4, 5, 2],
        "direction": [1] * 5,
        "cost": [1.0] * 5,
    }

    # Links 1 and 2 form the route 1 -> 2 -> 6. Links 3, 4 and 5 form
    # the only detour: 2 -> 4 -> 5 -> 2. The turn restriction bans link 2.
    turns = pd.DataFrame({"from_node": [1], "via_node": [2], "to_node": [6], "penalty": [np.inf]})
    graph = build_graph(links, [1, 6], cost="cost", skimming=["cost"], compact=True, turn_restrictions=turns)
    path = graph.compute_path(1, 6)
    np.testing.assert_array_equal(path.get_network_links()[0], [1, 3, 4, 5, 2])
    np.testing.assert_array_equal(path.path_nodes, [1, 2, 4, 5, 2, 6])
    assert path.milepost[-1] == 5.0

    # The cycle's links should stay separate, contracting it would create a 2 -> 2 loop.
    assert {4, 5}.issubset(graph.all_nodes)

    tails = np.repeat(np.arange(graph.num_nodes), np.diff(graph.routing.fs).astype(np.intp))
    assert not (tails == graph.routing.heads).any()
    assert (graph.mapping.network_link_to_graph >= 0).all()
    assert graph.compute_skims().results.skims["cost"][0, 1] == 5.0


def test_parallel_link_round_trip_is_not_a_turn_restriction_detour():
    links = {
        "link_id": [1, 2, 3, 4],
        "a_node": [1, 2, 2, 2],
        "b_node": [2, 6, 4, 4],
        "direction": [1, 1, 0, 0],
        "cost": [1.0] * 4,
    }
    turns = pd.DataFrame({"from_node": [1], "via_node": [2], "to_node": [6], "penalty": [np.inf]})
    graph = build_graph(links, [1, 6], cost="cost", skimming=["cost"], compact=True, turn_restrictions=turns)

    # The two parallel bidirectional links form 2 -> 4 -> 2. This returns to
    # the same previous node, so it is a U-turn, not a way around the ban.
    assert graph.compute_path(1, 6).path is None
    assert 4 in graph.all_nodes
    assert {3, 4}.issubset(graph.mapping.network.link_ids[graph.mapping.expansion_links])

    tails = np.repeat(np.arange(graph.num_nodes), np.diff(graph.routing.fs).astype(np.intp))
    assert not (tails == graph.routing.heads).any()
    assert np.isinf(graph.compute_skims().results.skims["cost"][0, 1])


def graph_with_required_self_loop():
    links = {
        "link_id": [1, 2, 3, 4],
        "a_node": [1, 2, 3, 3],
        "b_node": [2, 3, 4, 3],
        "direction": [1] * 4,
        "cost": [1.0, 1.0, 1.0, 2.0],
    }
    turns = pd.DataFrame({"from_node": [2], "via_node": [3], "to_node": [4], "penalty": [np.inf]})
    return build_graph(links, [1, 4], cost="cost", skimming=["cost"], compact=True, turn_restrictions=turns)


def test_compaction_keeps_real_self_loop_needed_to_avoid_turn_ban():
    graph = graph_with_required_self_loop()

    # Links 1 -> 2 -> 3 approach the turn. Link 3 exits to 4, but the
    # restriction bans 2 -> 3 -> 4. Link 4 is a genuine loop at node 3.
    path = graph.compute_path(1, 4)
    np.testing.assert_array_equal(path.get_network_links()[0], [1, 2, 4, 3])
    np.testing.assert_array_equal(path.path_nodes, [1, 3, 3, 4])
    assert path.milepost[-1] == 5.0
    assert 2 not in graph.all_nodes  # The separate chain 1 -> 2 -> 3 is compressed.
    tails = np.repeat(np.arange(graph.num_nodes), np.diff(graph.routing.fs).astype(np.intp))
    loops = np.flatnonzero(tails == graph.routing.heads)
    np.testing.assert_array_equal(graph.mapping.expand_links(loops)[0], [4])
    assert graph.compute_skims().results.skims["cost"][0, 1] == 5.0


def test_set_graph(sioux_falls_example):
    graph = sioux_falls_example.network.build_graph("c", "distance")
    assert graph.num_zones == 24, "Number of centroids not properly set"
    assert graph.num_links == 76, "Number of links not properly set"
    assert graph.num_nodes == 24, f"Number of nodes not properly set - {graph.num_nodes}"


def test_available_skims(sioux_falls_example):
    network = sioux_falls_example.network.build()
    avail = network.fields
    data_fields = ["distance", "name", "lanes", "capacity", "speed", "b", "free_flow_time", "power", "modes"]
    assert all(i in avail for i in data_fields), "Skim availability with problems"


def test_compute_path(sioux_falls_example):
    graph = sioux_falls_example.network.build().build_graph("c", cost="distance", block_centroid_flows=False)

    res = graph.compute_path(1, 6)
    assert list(res.get_network_links()[0]) == [1, 4], "Number of path links is not correct"
    assert list(res.path_nodes) == [1, 2, 6], "Number of path nodes is not correct"


def test_set_turn_restrictions_rejects_negative_penalty(sioux_falls_example):
    network = sioux_falls_example.network.build()
    graph = network.build_graph("c", cost="distance", block_centroid_flows=False)

    from_node, via_node, to_node = sample_turn_from_path(graph, 1, 6)
    turn_restrictions = pd.DataFrame(
        {"from_node": [from_node], "via_node": [via_node], "to_node": [to_node], "penalty": [-1.0]}
    )

    with pytest.raises(ValueError, match="Negative turn penalties"):
        network.build_graph("c", cost="distance", turn_restrictions=turn_restrictions)


@pytest.fixture
def turn_network():
    return Network(
        pd.DataFrame(
            {
                "link_id": [1, 2, 3],
                "a_node": [10, 20, 30],
                "b_node": [20, 30, 40],
                "direction": [1, 1, 1],
                "cost": [1.0, 1.0, 1.0],
            }
        )
    )


@pytest.mark.parametrize("turn", [[-999, 20, 30], [10, 15, 30], [10, 20, 999]])
def test_turn_restrictions_reject_missing_nodes(turn_network, turn):
    turns = pd.DataFrame(
        {"from_node": [turn[0]], "via_node": [turn[1]], "to_node": [turn[2]], "penalty": [5.0]}, index=[7]
    )
    with pytest.raises(ValueError, match="missing directed legs"):
        turn_network.build_graph("c", "cost", turn_restrictions=turns)


@pytest.mark.parametrize(
    "turn, leg",
    [
        ([20, 10, 20], "from_node -> via_node"),
        ([10, 20, 10], "via_node -> to_node"),
        ([10, 30, 40], "from_node -> via_node"),
        ([10, 20, 40], "via_node -> to_node"),
    ],
)
def test_turn_restrictions_reject_missing_directed_legs(turn_network, turn, leg):
    turns = pd.DataFrame(
        {"from_node": [turn[0]], "via_node": [turn[1]], "to_node": [turn[2]], "penalty": [5.0]}, index=[7]
    )
    with pytest.raises(ValueError, match=f"missing directed legs {leg}"):
        turn_network.build_graph("c", "cost", turn_restrictions=turns)


def test_dead_end_removal_preserves_turn_via_node():
    links = {
        "link_id": [1, 2, 3],
        "a_node": [1, 2, 2],
        "b_node": [2, 3, 4],
        "direction": [1, 0, 1],
        "cost": [1.0, 1.0, 1.0],
    }
    turns = pd.DataFrame({"from_node": [1, 2], "via_node": [2, 3], "to_node": [4, 2], "penalty": [np.inf, 5.0]})
    graph = build_graph(
        links,
        [1, 4],
        cost="cost",
        skimming=["cost"],
        compact=True,
        remove_dead_ends=True,
        turn_restrictions=turns,
        allow_uturns=True,
    )

    assert 3 in graph.all_nodes
    assert (graph.mapping.network_link_to_graph >= 0).all()

    path = graph.compute_path(1, 4)
    np.testing.assert_array_equal(path.path_nodes, [1, 2, 3, 2, 4])
    assert path.milepost[-1] == 9.0

    assert graph.compute_skims().results.skims["cost"][0, 1] == 9.0


def test_compaction_preserves_bidir_nodes_when_path_uturns_are_allowed():
    links = {
        "link_id": [1, 2, 3],
        "a_node": [10, 20, 30],
        "b_node": [20, 30, 40],
        "direction": [0, 0, 0],
        "cost": [1.0, 1.0, 1.0],
    }
    turns = pd.DataFrame({"from_node": [10], "via_node": [20], "to_node": [30], "penalty": [np.inf]})
    graph = build_graph(links, [10, 40], compact=True, turn_restrictions=turns, allow_uturns=True)
    # Node 20 is the restricted turn. Node 30 has no restriction but still
    # needs to remain explicit because U-turns are allowed there too.
    assert 20 in graph.all_nodes
    assert 30 in graph.all_nodes


def test_dead_end_removal_preserves_branch_needed_for_allowed_uturn():
    links = {
        "link_id": [1, 2, 3],
        "a_node": [4, 1, 1],
        "b_node": [1, 3, 2],
        "direction": [1, 1, 0],
        "cost": [1.0, 1.0, 1.0],
    }
    turns = pd.DataFrame({"from_node": [4], "via_node": [1], "to_node": [3], "penalty": [np.inf]})
    graph = build_graph(
        links,
        [4, 3],
        cost="cost",
        skimming=["cost"],
        compact=True,
        remove_dead_ends=True,
        turn_restrictions=turns,
        allow_uturns=True,
    )

    # The direct movement is banned. Reaching 3 requires entering the spur,
    # reversing at node 2, then taking the now-allowed movement at node 1.
    path = graph.compute_path(4, 3)
    np.testing.assert_array_equal(path.path_nodes, [4, 1, 2, 1, 3])
    np.testing.assert_array_equal(path.get_network_links()[0], [1, 3, 3, 2])
    assert path.milepost[-1] == 4.0
    assert (graph.mapping.network_link_to_graph >= 0).all()
    assert graph.compute_skims().results.skims["cost"][0, 1] == 4.0


def test_dead_end_removal_stops_at_turn_via_node():
    links = {
        "link_id": [1, 2, 3],
        "a_node": [1, 30, 40],
        "b_node": [2, 40, 2],
        "direction": [1, 1, 1],
    }
    turns = pd.DataFrame({"from_node": [30], "via_node": [40], "to_node": [2], "penalty": [5.0]})
    graph = build_graph(links, [1, 2], compact=True, remove_dead_ends=True, turn_restrictions=turns)

    assert 40 in graph.all_nodes
    network = graph.mapping.network
    removed = network.link_ids[graph.mapping.network_link_to_graph < 0]
    np.testing.assert_array_equal(removed, [2])
    assert graph.routing.turn_penalties.size == 0


@pytest.mark.parametrize("penalty", [5.0, np.inf])
def test_turn_restrictions_allow_compressed_neighbour_nodes(penalty):
    links = {
        "link_id": [1, 2, 3, 4],
        "a_node": [1, 2, 3, 4],
        "b_node": [2, 3, 4, 5],
        "direction": [0, 0, 0, 0],
        "cost": [1.0, 1.0, 1.0, 1.0],
    }
    turns = pd.DataFrame({"from_node": [2], "via_node": [3], "to_node": [4], "penalty": [penalty]})
    graph = build_graph(links, [1, 5], cost="cost", skimming=["cost"], compact=True, turn_restrictions=turns)

    assert 3 in graph.all_nodes
    assert 2 not in graph.all_nodes
    assert 4 not in graph.all_nodes

    explicit = graph.routing.turn_penalties
    assert np.count_nonzero(explicit == penalty) >= 1

    path = graph.compute_path(1, 5)
    if np.isfinite(penalty):
        assert path.milepost[-1] == 4.0 + penalty
    else:
        assert path.path is None

    skims = graph.compute_skims().results.skims["cost"]
    assert skims[0, 1] == 4.0 + penalty
    assert skims[1, 0] == 4.0


@pytest.mark.parametrize("turn", [[999, 20, 30], [30, 20, 30]])
def test_prepare_graph_rejects_invalid_existing_turns(turn_network, turn):
    turns = pd.DataFrame(
        {
            "from_node": [turn[0]],
            "via_node": [turn[1]],
            "to_node": [turn[2]],
            "penalty": [5.0],
        }
    )

    with pytest.raises(ValueError, match="Turn restrictions have missing directed legs"):
        Network(turn_network.links, turn_restrictions=turns)


def test_a_star_with_turn_restrictions(sioux_falls_example):
    network = sioux_falls_example.network.build()
    graph = network.build_graph("c", cost="distance", block_centroid_flows=False)
    from_node, via_node, to_node = sample_turn_from_path(graph, 1, 6)
    turn_restrictions = pd.DataFrame(
        {"from_node": [from_node], "via_node": [via_node], "to_node": [to_node], "penalty": [np.inf]}
    )
    graph = network.build_graph("c", cost="distance", turn_restrictions=turn_restrictions, block_centroid_flows=False)

    coordinates = network.lonlat
    scale = estimate_heuristic_scale(graph, coordinates, heuristic="haversine")
    result = graph.compute_path(
        1, 6, a_star=True, heuristic="haversine", heuristic_scale=scale, coordinates=coordinates
    )
    assert result.milepost[-1] == graph.compute_path(1, 6).milepost[-1]
    assert result.path_nodes[0] == 1
    assert result.path_nodes[-1] == 6


def test_compute_skims(sioux_falls_example):
    graph = sioux_falls_example.network.build().build_graph(
        "c", cost="distance", skimming=["distance", "free_flow_time"], block_centroid_flows=False
    )

    skm = graph.compute_skims()
    skims = skm.results.skims
    assert len(skims) == 2, "Number of matrices is not correct"
    assert set(skims) == {"distance", "free_flow_time"}, "Matrices names are not correct"


def test_exclude_links(sioux_falls_example):
    network = sioux_falls_example.network.build()
    graph = network.build_graph("c", cost="distance", block_centroid_flows=False)
    origin = 20
    destination = 21
    r1 = PathResults(graph, origin, destination)

    assert list(r1.get_network_links()[0]) == [62]
    costs = network.field_values("distance").copy()
    costs[network.link_ids == 62] = np.inf
    graph.set_costs(costs)
    r1 = PathResults(graph, origin, destination)
    assert list(r1.get_network_links()[0]) == [63, 69]


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
    for k, v in transit_graph.config.items():
        assert transit_graph_obj._config[k] == v


def test_transit_graph_od_node_mapping(transit_graph):
    transit_graph_obj = transit_graph.to_transit_graph()
    pd.testing.assert_frame_equal(transit_graph.od_node_mapping, transit_graph_obj.od_node_mapping)


@pytest.fixture(scope="function")
def compressed_graph(test_data_path, tmp_path):
    zipfile.ZipFile(test_data_path / "KaiTang.zip").extractall(tmp_path)

    link_df = pd.read_csv(tmp_path / "links_modified.csv")
    return Network(link_df, centroids=[7, 8, 11]).build_graph(
        "a", "fft", compact=True, remove_dead_ends=True, block_centroid_flows=False
    )


def test_compressed_graph(compressed_graph):
    # Check the compressed links, links 4 and 5 should be collapsed into 2 links from 3 - 10 and 10 - 3.
    mapping = compressed_graph.mapping
    contracted = np.flatnonzero(np.diff(mapping.expansion_offsets) > 1)
    link_ids, _ = mapping.expand_links(contracted)
    assert set(link_ids) == {4, 5}

    # Confirm these compacted links map back up to a contraction between the correct nodes.
    tails = np.repeat(np.arange(compressed_graph.num_nodes), np.diff(compressed_graph.routing.fs).astype(np.intp))
    endpoints = np.column_stack(
        (
            compressed_graph.all_nodes[tails[contracted]],
            compressed_graph.all_nodes[compressed_graph.routing.heads[contracted]],
        )
    )
    assert {tuple(pair) for pair in endpoints} == {(3, 10), (10, 3)}


def test_dead_end_removal(compressed_graph):
    # Dead-end removal removes the entire branch (links 30 through 38), not
    # just compressed representatives 30 and 38. Link 40 is a one-way sink,
    # retaining it cannot introduce a route back into the rest of the network.
    mapping = compressed_graph.mapping
    removed = mapping.network.link_ids[mapping.network_link_to_graph < 0]
    assert set(removed) == set(range(30, 39)), "Dead end removal removed incorrect links"


def test_turn_restrictions_match_networkx(coquimbo_example):
    nx = pytest.importorskip("networkx")

    network = coquimbo_example.network.build()
    graph = network.build_graph("c", "distance", block_centroid_flows=False)
    origin = 32343
    destination = 22041

    mapping = network.mapping
    graph_df = pd.DataFrame(
        {
            "id": np.arange(mapping.link_count),
            "a_node": mapping.node_ids[mapping.link_tails],
            "b_node": mapping.node_ids[mapping.link_heads],
            "link_id": mapping.link_ids,
            "direction": mapping.directions,
            "distance": network.field_values("distance"),
        }
    )
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
    _, base_cost_aeq = aeq_directed_path(res_before.path_nodes, res_before.get_network_links()[0])
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

    graph = network.build_graph("c", "distance", turn_restrictions=turn_restrictions, block_centroid_flows=False)
    res_after = graph.compute_path(origin, destination)
    directed_after_aeq, restricted_cost_aeq = aeq_directed_path(res_after.path_nodes, res_after.get_network_links()[0])
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
