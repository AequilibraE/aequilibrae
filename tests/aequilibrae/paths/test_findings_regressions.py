import numpy as np
import pandas as pd
import pytest

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae.paths.network_skimming import NetworkSkimming
from aequilibrae.paths.results import AssignmentResults


def _make_graph_from_network(
    links_data: list[dict],
    centroids: list[int],
    modes: str = "c",
) -> Graph:
    df = pd.DataFrame(links_data)
    df["modes"] = modes
    df["link_type"] = "road"
    graph = Graph()
    graph.cost_field = "cost"
    graph.network = df
    graph.prepare_graph(
        centroids=np.array(centroids, dtype=np.int64),
        remove_dead_ends=True,
    )
    graph.set_graph("cost")
    return graph


def test_finding_1_finite_uturn_spur_not_burned():
    """Finding 1A: Dead-end removal must not burn turn-via spurs or turnaround spurs when U-turns are enabled."""
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 3, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 2, "a_node": 3, "b_node": 2, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 3, "a_node": 3, "b_node": 4, "direction": 0, "distance": 1.0, "cost": 1.0},
    ]
    turns = pd.DataFrame(
        [
            {"from_node": 1, "via_node": 3, "to_node": 2, "penalty": None},
            {"from_node": 3, "via_node": 4, "to_node": 3, "penalty": 2.0},
        ]
    )

    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"
    graph = Graph()
    graph.cost_field = "cost"
    graph.network = df
    graph.set_turn_restrictions(turns, allow_path_uturns=False)
    graph.prepare_graph(
        centroids=np.array([1, 2], dtype=np.int64),
        remove_dead_ends=True,
    )
    graph.set_graph("cost")

    # Spur link 3 must not be in dead_end_links
    assert 3 not in graph.dead_end_links

    # Path 1 -> 3 -> 4 -> 3 -> 2 must be reachable
    result = graph.compute_path(1, 2)
    assert result.path is not None
    assert list(result.path_nodes) == [1, 3, 4, 3, 2]
    # Expected cost: 1 (link 1) + 1 (link 3 AB) + 2.0 (turn at 4) + 1 (link 3 BA) + 1 (link 2) = 6.0
    assert result.milepost[-1] == pytest.approx(6.0)

    # Compact network skimming must also find it reachable with cost 6.0
    graph.set_skimming("cost")
    skimmer = NetworkSkimming(graph)
    skimmer.execute()
    matrix = skimmer.results.skims
    origin_pos = int(np.flatnonzero(matrix.index == 1)[0])
    dest_pos = int(np.flatnonzero(matrix.index == 2)[0])
    assert matrix.cost[origin_pos, dest_pos] == pytest.approx(6.0)


def test_finding_1_chain_compression_self_loop_cycle_retention():
    """Finding 1B: Chain compression must not remove original cycle links when dropping self-loops."""
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 3, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 2, "a_node": 3, "b_node": 2, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 3, "a_node": 3, "b_node": 4, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 4, "a_node": 4, "b_node": 5, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 5, "a_node": 5, "b_node": 3, "direction": 1, "distance": 1.0, "cost": 1.0},
    ]
    # Prohibit direct 1 -> 3 -> 2, forcing vehicle to take the cycle 3 -> 4 -> 5 -> 3 then turn 5 -> 3 -> 2
    turns = pd.DataFrame(
        [
            {"from_node": 1, "via_node": 3, "to_node": 2, "penalty": None},
        ]
    )

    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"
    graph = Graph()
    graph.cost_field = "cost"
    graph.network = df
    graph.set_turn_restrictions(turns, allow_path_uturns=False)
    graph.prepare_graph(
        centroids=np.array([1, 2], dtype=np.int64),
        remove_dead_ends=False,
    )
    graph.set_graph("cost")

    # Original links 3, 4, 5 must NOT be removed from compact graph!
    compact_link_ids = set(graph.compact_graph.link_id.values)
    assert {3, 4, 5}.issubset(compact_link_ids) or any(lid >= 10 for lid in compact_link_ids)

    # Path from 1 to 2 must be reachable
    result = graph.compute_path(1, 2)
    assert result.path is not None
    assert list(result.path_nodes) == [1, 3, 4, 5, 3, 2]
    # Cost: 1.0 + 1.0 + 1.0 + 1.0 + 1.0 = 5.0
    assert result.milepost[-1] == pytest.approx(5.0)

    # Skimming must report 5.0
    graph.set_skimming("cost")
    skimmer = NetworkSkimming(graph)
    skimmer.execute()
    matrix = skimmer.results.skims
    orig_idx = int(np.flatnonzero(matrix.index == 1)[0])
    dest_idx = int(np.flatnonzero(matrix.index == 2)[0])
    assert matrix.cost[orig_idx, dest_idx] == pytest.approx(5.0)


def test_finding_2_turn_aware_skimming_dual_arrivals():
    """Finding 2: Turn-aware skimming must backtrack arc state rather than collapsing to cheaper arrivals."""
    # Route Cheap: 1 -> 4 -> 3 (cost 1 + 1 = 2)
    # Route Detour: 1 -> 5 -> 3 (cost 5 + 5 = 10)
    # Exit: 3 -> 2 (cost 3)
    # Turn 4 -> 3 -> 2 is prohibited (None)
    # Turn 5 -> 3 -> 2 is allowed with penalty 1.0
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 4, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 2, "a_node": 4, "b_node": 3, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 3, "a_node": 1, "b_node": 5, "direction": 1, "distance": 5.0, "cost": 5.0},
        {"link_id": 4, "a_node": 5, "b_node": 3, "direction": 1, "distance": 5.0, "cost": 5.0},
        {"link_id": 5, "a_node": 3, "b_node": 2, "direction": 1, "distance": 3.0, "cost": 3.0},
    ]
    turns = pd.DataFrame(
        [
            {"from_node": 4, "via_node": 3, "to_node": 2, "penalty": None},
            {"from_node": 5, "via_node": 3, "to_node": 2, "penalty": 1.0},
        ]
    )

    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"
    graph = Graph()
    graph.cost_field = "cost"
    graph.network = df
    graph.set_turn_restrictions(turns, allow_path_uturns=False)
    graph.prepare_graph(
        centroids=np.array([1, 2], dtype=np.int64),
        remove_dead_ends=False,
    )
    graph.set_graph("cost")

    result = graph.compute_path(1, 2)
    assert result.path is not None
    assert list(result.path_nodes) == [1, 5, 3, 2]
    # Expected cost: 5.0 + 5.0 + 1.0 (turn penalty) + 3.0 = 14.0
    assert result.milepost[-1] == pytest.approx(14.0)

    graph.set_skimming("cost")
    skimmer = NetworkSkimming(graph)
    skimmer.execute()
    matrix = skimmer.results.skims
    orig_idx = int(np.flatnonzero(matrix.index == 1)[0])
    dest_idx = int(np.flatnonzero(matrix.index == 2)[0])
    # Skim must match chosen arc path (14.0), not collapsed node predecessor (2.0 + 3.0 = 5.0)
    assert matrix.cost[orig_idx, dest_idx] == pytest.approx(14.0)


def test_finding_3_empty_network_total_exclusion_and_zero_roots(tmp_path):
    """Finding 3: Zero-root and total exclusion must not crash with IndexError or leave stale state."""
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "distance": 1.0, "cost": 1.0},
    ]
    graph = _make_graph_from_network(links, centroids=[1, 2])
    assert graph.num_links > 0

    # Exclude all links
    graph.exclude_links([1])
    assert graph.network.empty
    assert graph.num_links == 0
    assert graph.compact_num_links == 0
    assert len(graph.cost) == 0

    # Preparing an empty network (0 roots, empty graph) must not crash
    empty_df = pd.DataFrame(
        columns=["link_id", "a_node", "b_node", "direction", "distance", "cost", "modes", "link_type"]
    )
    empty_graph = Graph()
    empty_graph.cost_field = "cost"
    empty_graph.network = empty_df
    empty_graph.prepare_graph(centroids=np.array([1, 2], dtype=np.int64))
    assert empty_graph.num_links == 0
    assert empty_graph.compact_num_links == 0

    # Total exclusion of every link in a prepared graph
    links2 = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 2, "a_node": 2, "b_node": 3, "direction": 1, "distance": 1.0, "cost": 1.0},
    ]
    graph_excluded = _make_graph_from_network(links2, centroids=[1, 3])
    assert graph_excluded.num_links > 0
    graph_excluded.exclude_links([1, 2])
    assert graph_excluded.network.empty
    assert graph_excluded.num_links == 0
    assert graph_excluded.compact_num_links == 0

    # 1. NetworkSkimming on totally excluded graph must not fail with IndexError
    graph_excluded.set_skimming("cost")
    skimmer = NetworkSkimming(graph_excluded)
    skimmer.execute()
    matrix = skimmer.results.skims
    assert np.all(np.isnan(matrix.cost))

    # 2. Assignment on a totally excluded graph
    mat = AequilibraeMatrix()
    mat.create_empty(zones=2, matrix_names=["demand"])
    mat.matrix["demand"][:, :] = 1.0
    mat.index[:] = [1, 3]
    mat.computational_view()

    res = AssignmentResults()
    res.prepare(graph_excluded, mat)
    aon = allOrNothing("demand", mat, graph_excluded, res)
    aon.execute()
    # The mode has no active arcs, but assignment result vectors remain in the
    # original project-wide supernet index space.
    assert res.link_loads.shape == (graph_excluded.supernet_size, 1)

    # 3. Save to disk and load from disk on excluded graph (and verify fresh UUID is kept)
    save_file = str(tmp_path / "excluded_graph.aeq")
    graph_excluded.save_to_disk(save_file)
    loaded_graph = Graph()
    loaded_graph.load_from_disk(save_file)
    assert loaded_graph._id != graph_excluded._id
    assert loaded_graph.num_links == 0
    assert loaded_graph.compact_num_links == 0

    # 4. Also verify loaded non-empty graph preserves fresh UUID (Issue 6)
    g_normal = _make_graph_from_network(links2, centroids=[1, 3])
    normal_save_file = str(tmp_path / "normal_graph.aeq")
    g_normal.save_to_disk(normal_save_file)
    loaded_normal = Graph()
    loaded_normal.load_from_disk(normal_save_file)
    assert loaded_normal._id != g_normal._id


def test_finding_6_effective_turn_topology_signatures():
    """Finding 6: Turn topology signature must only consider mappable turn restrictions."""
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "distance": 1.0, "cost": 1.0},
        {"link_id": 2, "a_node": 2, "b_node": 3, "direction": 1, "distance": 1.0, "cost": 1.0},
    ]
    # Phantom turn restriction between nodes that don't have edges in the graph
    turns = pd.DataFrame(
        [
            {"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 1.0},
            {"from_node": 999, "via_node": 888, "to_node": 777, "penalty": 2.0},
        ]
    )

    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"
    graph = Graph()
    graph.cost_field = "cost"
    graph.network = df
    graph.set_turn_restrictions(turns)
    graph.prepare_graph(centroids=np.array([1, 3], dtype=np.int64))

    effective_vias = graph._compute_effective_turn_vias()
    assert 2 in effective_vias
    assert 888 not in effective_vias

    sig = graph._compute_turn_topology_signature()
    assert sig[0] == (2,)


def test_arc_based_skimming_memoized_prefix_tree():
    """Finding 5 (arc-based skimming): Linear-time memoized prefix tree accumulation in skim_arc_based_paths."""
    links = [
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "distance": 10.0, "cost": 10.0},
        {"link_id": 2, "a_node": 2, "b_node": 3, "direction": 1, "distance": 5.0, "cost": 5.0},
        {"link_id": 3, "a_node": 2, "b_node": 4, "direction": 1, "distance": 8.0, "cost": 8.0},
        {"link_id": 4, "a_node": 3, "b_node": 5, "direction": 1, "distance": 3.0, "cost": 3.0},
        {"link_id": 5, "a_node": 4, "b_node": 6, "direction": 1, "distance": 4.0, "cost": 4.0},
        {"link_id": 6, "a_node": 3, "b_node": 7, "direction": 1, "distance": 6.0, "cost": 6.0},
    ]
    turns = pd.DataFrame(
        [
            {"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 2.0},
            {"from_node": 1, "via_node": 2, "to_node": 4, "penalty": 4.0},
            {"from_node": 2, "via_node": 3, "to_node": 5, "penalty": 1.0},
            {"from_node": 2, "via_node": 4, "to_node": 6, "penalty": 1.5},
            {"from_node": 2, "via_node": 3, "to_node": 7, "penalty": 0.5},
        ]
    )
    # Centroids are 1 (origin), and 5, 6, 7 (destinations). Nodes 2, 3, 4 are transit nodes.
    centroids = np.array([1, 5, 6, 7], dtype=np.int64)

    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"
    graph = Graph()
    graph.cost_field = "cost"
    graph.network = df
    graph.set_turn_restrictions(turns, allow_path_uturns=False)
    graph.prepare_graph(centroids=centroids, remove_dead_ends=False)
    graph.set_graph("cost")
    graph.set_skimming(["cost", "distance"])

    skimmer = NetworkSkimming(graph)
    skimmer.execute()
    skims = skimmer.results.skims

    idx_map = {c: int(np.flatnonzero(skims.index == c)[0]) for c in centroids}
    o_idx = idx_map[1]

    # Destination 5:
    # Path: 1 -> 2 -> 3 -> 5
    # link costs = 10 + 5 + 3 = 18.0; turns = 2.0 + 1.0 = 3.0 -> total cost = 21.0
    # link distance = 10 + 5 + 3 = 18.0
    assert skims.cost[o_idx, idx_map[5]] == pytest.approx(21.0)
    assert skims.distance[o_idx, idx_map[5]] == pytest.approx(18.0)

    # Destination 6:
    # Path: 1 -> 2 -> 4 -> 6
    # link costs = 10 + 8 + 4 = 22.0; turns = 4.0 + 1.5 = 5.5 -> total cost = 27.5
    # link distance = 10 + 8 + 4 = 22.0
    assert skims.cost[o_idx, idx_map[6]] == pytest.approx(27.5)
    assert skims.distance[o_idx, idx_map[6]] == pytest.approx(22.0)

    # Destination 7:
    # Path: 1 -> 2 -> 3 -> 7 (shares prefix 1 -> 2 -> 3 with destination 5)
    # link costs = 10 + 5 + 6 = 21.0; turns = 2.0 + 0.5 = 2.5 -> total cost = 23.5
    # link distance = 10 + 5 + 6 = 21.0
    assert skims.cost[o_idx, idx_map[7]] == pytest.approx(23.5)
    assert skims.distance[o_idx, idx_map[7]] == pytest.approx(21.0)
