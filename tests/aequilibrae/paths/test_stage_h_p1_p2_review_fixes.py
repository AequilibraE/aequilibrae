import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from aequilibrae.paths import Graph
from aequilibrae.paths.traffic_class import TrafficClass
from aequilibrae.paths.traffic_assignment import TrafficAssignment
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths.cython.basic_path_finding import path_finding_hybrid


def make_simple_network():
    # Linear graph: 1 -> 2 -> 3 -> 4
    df = pd.DataFrame(
        {
            "link_id": [1, 2, 3],
            "a_node": [1, 2, 3],
            "b_node": [2, 3, 4],
            "direction": [1, 1, 1],
            "distance": [10.0, 10.0, 10.0],
            "free_flow_time": [1.0, 1.0, 1.0],
            "capacity": [1000.0, 1000.0, 1000.0],
        }
    )
    return df


def test_issue_1_trailing_isolated_centroids():
    """Isolated centroids can index beyond the degree arrays in pruning."""
    net = make_simple_network()
    # Centroids with node IDs much higher than any link node ID
    centroids = np.array([1, 50, 100], dtype=np.int64)

    # Test with pruning enabled
    g1 = Graph()
    g1.network = net.copy()
    g1.prepare_graph(centroids=centroids, remove_dead_ends=True)
    g1.set_graph("distance")
    assert g1.num_nodes >= 100 or g1.nodes_to_indices.shape[0] > 100
    assert g1.compact_num_nodes == 3

    # Test with pruning disabled
    g2 = Graph()
    g2.network = net.copy()
    g2.prepare_graph(centroids=centroids, remove_dead_ends=False)
    g2.set_graph("distance")
    assert g2.compact_num_nodes == 4


def test_issue_2_empty_compact_graph_crosswalk_and_reprepare():
    """Empty compact graphs must initialize full-to-compact crosswalk and clear boundary state."""
    # A graph where dead-end pruning removes every single link:
    # Nodes 3 -> 4 -> 5 are not centroids (centroids are 1, 2), so dead-end pruning burns all of them!
    net = pd.DataFrame(
        {
            "link_id": [1, 2],
            "a_node": [3, 4],
            "b_node": [4, 5],
            "direction": [1, 1],
            "distance": [10.0, 10.0],
            "free_flow_time": [1.0, 1.0],
            "capacity": [1000.0, 1000.0],
        }
    )
    centroids = np.array([1, 2], dtype=np.int64)
    g = Graph()
    g.network = net
    g.prepare_graph(centroids=centroids, remove_dead_ends=True)
    g.set_graph("distance")

    assert g.compact_num_links == 0
    assert (g.graph["__compressed_id__"] == 0).all()
    assert (g._crosswalk == 0).all()
    assert len(g._compact_first_node) == 0
    assert len(g._compact_last_node) == 0

    # Test repreparation does not crash or leave stale arrays
    g.prepare_graph(centroids=centroids, remove_dead_ends=True)
    assert g.compact_num_links == 0


def test_compact_crosswalk_uses_project_wide_supernet_size_and_rebuilds_when_enlarged():
    """Mode-local arcs retain their positions in the project-wide cost vector."""
    net = pd.DataFrame(
        {
            "link_id": [1, 2],
            "a_node": [1, 2],
            "b_node": [2, 3],
            "direction": [1, 1],
            "distance": [1.0, 1.0],
            "__supernet_id_ab": [1, 4],
            "__supernet_id_ba": [-1, -1],
        }
    )
    g = Graph()
    g.supernet_size = 7
    g.network = net
    g.prepare_graph(centroids=np.array([1, 3], dtype=np.int64), remove_dead_ends=False)
    g.set_graph("distance")

    assert len(g._crosswalk) == 7
    assert g._crosswalk[1] < g.compact_num_links
    assert g._crosswalk[4] < g.compact_num_links
    assert np.all(g._crosswalk[[0, 2, 3, 5, 6]] == g.compact_num_links)

    # Assignment can enlarge the shared index after the graph was prepared.  A
    # subsequent cost update must detect and replace the stale crosswalk.
    g.supernet_size = 9
    costs = np.zeros(9, dtype=np.float64)
    costs[[1, 4]] = [2.0, 3.0]
    g.compact_costs_from_link_costs(costs)
    assert len(g._crosswalk) == 9
    assert g.compact_cost[0] == pytest.approx(5.0)


def test_all_pruned_crosswalk_retains_project_wide_supernet_size():
    net = pd.DataFrame(
        {
            "link_id": [1, 2],
            "a_node": [3, 4],
            "b_node": [4, 5],
            "direction": [1, 1],
            "distance": [1.0, 1.0],
            "__supernet_id_ab": [1, 4],
            "__supernet_id_ba": [-1, -1],
        }
    )
    g = Graph()
    g.supernet_size = 7
    g.network = net
    g.prepare_graph(centroids=np.array([1, 2], dtype=np.int64), remove_dead_ends=True)

    assert g.compact_num_links == 0
    assert len(g._crosswalk) == 7
    assert np.all(g._crosswalk == g.compact_num_links)


def test_empty_graph_preserves_and_round_trips_project_wide_supernet_size(tmp_path):
    net = pd.DataFrame(
        columns=[
            "link_id",
            "a_node",
            "b_node",
            "direction",
            "distance",
            "__supernet_id_ab",
            "__supernet_id_ba",
        ]
    )
    g = Graph()
    g.supernet_size = 7
    g.network = net
    g.prepare_graph(centroids=np.array([1, 2], dtype=np.int64))

    assert g.supernet_size == 7
    assert g.num_links == 0
    assert len(g._crosswalk) == 7

    graph_file = tmp_path / "empty-global-supernet.aeg"
    g.save_to_disk(graph_file)
    loaded = Graph()
    loaded.load_from_disk(graph_file)

    assert loaded.supernet_size == 7
    assert loaded.num_links == 0
    assert loaded.compact_num_links == 0
    assert loaded.num_nodes == 2
    assert loaded.fs.shape == (3,)
    assert len(loaded._crosswalk) == 7


def test_mode_graph_round_trip_preserves_global_supernet_ids(tmp_path):
    net = pd.DataFrame(
        {
            "link_id": [1, 2],
            "a_node": [1, 2],
            "b_node": [2, 3],
            "direction": [1, 1],
            "distance": [1.0, 1.0],
            "__supernet_id_ab": [1, 4],
            "__supernet_id_ba": [-1, -1],
        }
    )
    g = Graph()
    g.supernet_size = 7
    g.network = net
    g.prepare_graph(centroids=np.array([1, 3], dtype=np.int64), remove_dead_ends=False)
    expected_ids = g.graph["__supernet_id__"].to_numpy(copy=True)

    graph_file = tmp_path / "mode-global-supernet.aeg"
    g.save_to_disk(graph_file)
    loaded = Graph()
    loaded.load_from_disk(graph_file)

    assert loaded.supernet_size == 7
    np.testing.assert_array_equal(loaded.graph["__supernet_id__"], expected_ids)
    assert len(loaded._crosswalk) == 7


def test_issue_3_genuine_original_self_loops_preserved():
    """Contraction must not delete genuine source-network self-loops unrelated to chains."""
    # Network with chain 1 -> 2 -> 3 and an uncontracted self-loop 4 -> 4
    net = pd.DataFrame(
        {
            "link_id": [1, 2, 3],
            "a_node": [1, 2, 4],
            "b_node": [2, 3, 4],
            "direction": [1, 1, 1],
            "distance": [10.0, 10.0, 5.0],
            "free_flow_time": [1.0, 1.0, 0.5],
            "capacity": [1000.0, 1000.0, 1000.0],
        }
    )
    centroids = np.array([1, 3, 4], dtype=np.int64)
    g = Graph()
    g.network = net
    g.prepare_graph(centroids=centroids, remove_dead_ends=False)
    g.set_graph("distance")

    # Links 1 and 2 compress into 1 compact link. Self-loop 3 (4 -> 4) must be preserved!
    assert (g.compact_graph["a_node"] == g.compact_graph["b_node"]).any()
    self_loop_compact = g.compact_graph[g.compact_graph["a_node"] == g.compact_graph["b_node"]]
    assert len(self_loop_compact) == 1
    compact_node_idx = self_loop_compact.iloc[0]["a_node"]
    assert g.compact_all_nodes[compact_node_idx] == 4


def test_issue_4_effective_vias_cache_invalidation_on_same_size_replacement():
    """Effective-turn-via cache must be invalidated on same-size topology replacement."""
    g = Graph()
    # Network 1: 4 links, node 2 is NOT connected to 3 (so 1->2->3 is ineffective)
    net1 = pd.DataFrame(
        {
            "link_id": [1, 2, 3, 4],
            "a_node": [1, 2, 5, 6],
            "b_node": [2, 4, 6, 7],
            "direction": [1, 1, 1, 1],
            "distance": [10.0, 10.0, 10.0, 10.0],
            "free_flow_time": [1.0, 1.0, 1.0, 1.0],
            "capacity": [1000.0, 1000.0, 1000.0, 1000.0],
        }
    )
    # Turn restriction on 1 -> 2 -> 3
    turns = pd.DataFrame(
        {
            "from_node": [1],
            "via_node": [2],
            "to_node": [3],
            "penalty": [15.0],
        }
    )
    g.network = net1
    g.set_turn_restrictions(turns)
    g.prepare_graph(centroids=np.array([1, 4, 5, 7], dtype=np.int64), remove_dead_ends=False)
    assert 2 not in g._compute_effective_turn_vias()

    # Network 2: exactly 4 links, but 1 -> 2 -> 3 -> 4 now connects 2 to 3!
    net2 = pd.DataFrame(
        {
            "link_id": [1, 2, 3, 4],
            "a_node": [1, 2, 3, 5],
            "b_node": [2, 3, 4, 6],
            "direction": [1, 1, 1, 1],
            "distance": [10.0, 10.0, 10.0, 10.0],
            "free_flow_time": [1.0, 1.0, 1.0, 1.0],
            "capacity": [1000.0, 1000.0, 1000.0, 1000.0],
        }
    )
    g.network = net2
    g.prepare_graph(centroids=np.array([1, 4, 5, 6], dtype=np.int64), remove_dead_ends=False)
    # Node 2 must now be recognized as an effective via!
    assert 2 in g._compute_effective_turn_vias()


def test_issue_5_skim_congested_compact_costs_under_chains():
    """skim_congested must refresh compact costs unconditionally whenever compressed."""
    net = make_simple_network()
    centroids = np.array([1, 4], dtype=np.int64)

    g = Graph()
    g.network = net
    g.prepare_graph(centroids=centroids, remove_dead_ends=False)
    g.set_graph("distance")
    g.set_skimming(["distance"])

    # Create dummy matrix
    mat = AequilibraeMatrix()
    mat.create_empty(memory_only=True, zones=2, matrix_names=["demand"])
    mat.index[:] = centroids
    mat.computational_view(core_list=["demand"])
    mat.matrix_view[:, :] = 100.0

    tc1 = TrafficClass("c1", g, mat)
    tc1.fixed_cost = np.array([10.0, 10.0, 10.0])
    tc1.congested_time = np.array([1.0, 1.0, 1.0])

    tc2 = TrafficClass("c2", g, mat)
    tc2.fixed_cost = np.array([100.0, 100.0, 100.0])
    tc2.congested_time = np.array([5.0, 5.0, 5.0])

    # Skim congested for tc1
    skm1 = tc1.skim_congested()
    c1_cost = float(skm1.results.skims.__assignment_cost__[0, 1])

    # Skim congested for tc2
    skm2 = tc2.skim_congested()
    c2_cost = float(skm2.results.skims.__assignment_cost__[0, 1])

    assert c2_cost > c1_cost
    assert np.isclose(c1_cost, 33.0)  # (10+1) * 3
    assert np.isclose(c2_cost, 315.0)  # (100+5) * 3
    assert float(g.compact_cost[0]) == 30.0


def test_issue_6_project_graph_turn_table_validation(sioux_falls_example):
    """Project graph construction validates modes column and applies set_turn_restrictions."""
    with sioux_falls_example.db_connection as conn:
        conn.execute("DROP TABLE IF EXISTS turn_restrictions")
        conn.execute("CREATE TABLE turn_restrictions (from_node INT, via_node INT, to_node INT, penalty REAL)")
        conn.execute("INSERT INTO turn_restrictions VALUES (1, 2, 3, 10.0)")
        conn.commit()

    with pytest.raises(ValueError, match="modes.*project.upgrade"):
        sioux_falls_example.network.build_graphs()


def test_issue_7_set_graph_cost_contract():
    """set_graph rejects negative values and -inf, coerces NaN to +inf, and keeps +inf."""
    net = make_simple_network()
    net["nan_cost"] = [1.0, np.nan, 3.0]
    net["neg_cost"] = [1.0, -2.0, 3.0]
    net["neginf_cost"] = [1.0, -np.inf, 3.0]
    net["posinf_cost"] = [1.0, np.inf, 3.0]

    g = Graph()
    g.network = net
    g.prepare_graph(centroids=np.array([1, 4], dtype=np.int64), remove_dead_ends=False)

    # NaN means "no data for this field" in real networks - the Coquimbo example ships a
    # travel_time column like that - so it is coerced to +inf (unusable) rather than rejected.
    g.set_graph("nan_cost")
    assert np.isinf(g.cost[1])

    with pytest.raises(ValueError, match="negative"):
        g.set_graph("neg_cost")

    with pytest.raises(ValueError, match="-inf"):
        g.set_graph("neginf_cost")

    # posinf_cost (+inf) must be accepted
    g.set_graph("posinf_cost")
    assert np.isinf(g.cost[1])
    assert np.isinf(g.compact_cost[0])


def test_issue_8_hybrid_origin_destination_handling():
    """Sparse hybrid search consumes origin in destination mask and returns 0 settled labels."""
    net = make_simple_network()
    centroids = np.array([1, 4], dtype=np.int64)
    g = Graph()
    g.network = net
    g.prepare_graph(centroids=centroids, remove_dead_ends=False)
    g.set_graph("distance")

    num_nodes = g.num_nodes
    num_arcs = g.num_links
    csr_indices = g.graph["b_node"].to_numpy(np.int64, copy=False)
    a_nodes = g.graph["a_node"].to_numpy(np.int64, copy=False)
    graph_costs = g.cost.astype(np.float64)
    graph_fs = g.fs.astype(np.int64)
    first_ctx = csr_indices
    last_ctx = a_nodes
    turn_fs = g.turn_fs
    turn_to_arcs = g.turn_to_arcs
    turn_penalties = g.turn_penalties
    stateful = g.stateful
    rep_arc = g.rep_arc

    destinations = np.zeros(num_nodes, dtype=np.uint8)
    destinations[0] = 1  # origin node index
    settled = np.zeros(1, dtype=np.int64)

    node_pred = np.full(num_nodes, -1, dtype=np.int64)
    connectors = np.full(num_nodes, -1, dtype=np.int64)
    reached_first = np.zeros(num_nodes, dtype=np.int64)
    node_costs = np.full(num_nodes, np.inf, dtype=np.float64)
    node_turn_penalties = np.zeros(num_nodes, dtype=np.float64)
    arc_pred = np.full(num_arcs, -1, dtype=np.int64)
    arc_turn_penalties = np.zeros(num_arcs, dtype=np.float64)

    found = path_finding_hybrid(
        0,  # origin
        destinations,
        1,  # destination_count = 1
        graph_costs,
        csr_indices,
        graph_fs,
        a_nodes,
        stateful,
        rep_arc,
        node_pred,
        connectors,
        reached_first,
        node_costs,
        node_turn_penalties,
        arc_pred,
        arc_turn_penalties,
        turn_fs,
        turn_to_arcs,
        turn_penalties,
        False,  # allow_uturns
        True,  # block_centroid_flows
        2,  # num_zones
        first_ctx,
        last_ctx,
        settled,
    )
    # The return value counts settled nodes excluding the origin, which is seeded into
    # reached_first[0] before the search starts - so consuming only the origin reports 0,
    # exactly as this test's name says. Every other exit of the kernel returns found - 1.
    assert found == 0
    assert settled[0] == 0


def test_issue_9_boolean_configuration_type_validation():
    """Configuration methods strictly require bool and raise TypeError on coercion attempts."""
    g = Graph()
    g.network = make_simple_network()

    with pytest.raises(TypeError, match="use_hybrid"):
        g.set_hybrid_kernel("False")

    with pytest.raises(TypeError, match="use_hybrid"):
        g.set_hybrid_kernel(0)

    with pytest.raises(TypeError, match="allow_path_uturns"):
        g.set_turn_restrictions(
            pd.DataFrame(columns=["from_node", "via_node", "to_node", "penalty"]), allow_path_uturns="True"
        )

    with pytest.raises(TypeError, match="remove_dead_ends"):
        g.prepare_graph(remove_dead_ends="False")

    with pytest.raises(TypeError, match="allow_uturns_everywhere"):
        g.prepare_graph(allow_uturns_everywhere=1)


def test_issue_10_use_hybrid_serialization_round_trip():
    """use_hybrid setting is persisted and restored across save_to_disk and load_from_disk."""
    net = make_simple_network()
    centroids = np.array([1, 4], dtype=np.int64)
    g = Graph()
    g.network = net
    g.prepare_graph(centroids=centroids, remove_dead_ends=False)
    g.set_hybrid_kernel(False)
    assert g.use_hybrid is False

    with tempfile.NamedTemporaryFile(suffix=".aeg", delete=False) as f:
        path = f.name
    try:
        g.save_to_disk(path)
        g2 = Graph()
        g2.load_from_disk(path)
        assert g2.use_hybrid is False
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_issue_11_conflicting_turn_penalties_rejected():
    """Conflicting finite turn penalties are rejected; prohibitions dominate finite entries."""
    g = Graph()
    g.network = make_simple_network()

    # Conflicting finite penalties for movement 1 -> 2 -> 3
    conflicting_df = pd.DataFrame(
        {
            "from_node": [1, 1],
            "via_node": [2, 2],
            "to_node": [3, 3],
            "penalty": [10.0, 20.0],
        }
    )
    with pytest.raises(ValueError, match="Conflicting duplicate turn penalties"):
        g.set_turn_restrictions(conflicting_df)

    # Identical penalties deduplicate cleanly
    identical_df = pd.DataFrame(
        {
            "from_node": [1, 1],
            "via_node": [2, 2],
            "to_node": [3, 3],
            "penalty": [15.0, 15.0],
        }
    )
    g.set_turn_restrictions(identical_df)
    assert len(g._turn_restrictions) == 1
    assert g._turn_restrictions.iloc[0]["penalty"] == 15.0

    # Prohibition dominates finite penalty
    prohib_df = pd.DataFrame(
        {
            "from_node": [1, 1],
            "via_node": [2, 2],
            "to_node": [3, 3],
            "penalty": [15.0, np.inf],
        }
    )
    g.set_turn_restrictions(prohib_df)
    assert len(g._turn_restrictions) == 1
    assert np.isinf(g._turn_restrictions.iloc[0]["penalty"])


def test_issue_12_per_class_turn_penalty_accounting():
    """c.results.total_turn_penalty tracks per-class turn penalties in AoN and equilibrium."""
    # Triangular network: 1 -> 2 -> 3 (distance 10, 10) and direct 1 -> 3 (distance 25)
    net = pd.DataFrame(
        {
            "link_id": [1, 2, 3],
            "a_node": [1, 2, 1],
            "b_node": [2, 3, 3],
            "direction": [1, 1, 1],
            "distance": [10.0, 10.0, 25.0],
            "free_flow_time": [10.0, 10.0, 25.0],
            "capacity": [100.0, 100.0, 100.0],
        }
    )
    turns = pd.DataFrame(
        {
            "from_node": [1],
            "via_node": [2],
            "to_node": [3],
            "penalty": [2.0],
        }
    )
    centroids = np.array([1, 3], dtype=np.int64)
    g = Graph()
    g.network = net
    g.set_turn_restrictions(turns)
    g.prepare_graph(centroids=centroids, remove_dead_ends=False)
    g.set_graph("free_flow_time")

    mat = AequilibraeMatrix()
    mat.create_empty(memory_only=True, zones=2, matrix_names=["demand"])
    mat.index[:] = centroids
    mat.computational_view(core_list=["demand"])
    mat.matrix_view[:, :] = 0.0
    mat.matrix_view[0, 1] = 50.0  # 50 units from 1 to 3

    assig = TrafficAssignment()
    assig.set_classes([TrafficClass("car", g, mat)])
    assig.set_vdf("BPR")
    assig.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assig.set_capacity_field("capacity")
    assig.set_time_field("free_flow_time")
    assig.set_algorithm("all-or-nothing")
    assig.execute()

    tc = assig.classes[0]
    # In AoN, all 50 units go 1 -> 2 -> 3, accumulating 50 * 2.0 = 100.0 turn penalty
    assert np.isclose(tc.results.total_turn_penalty, 100.0)

    # In FW equilibrium:
    assig_fw = TrafficAssignment()
    assig_fw.set_classes([TrafficClass("car", g, mat)])
    assig_fw.set_vdf("BPR")
    assig_fw.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assig_fw.set_capacity_field("capacity")
    assig_fw.set_time_field("free_flow_time")
    assig_fw.set_algorithm("frank-wolfe")
    assig_fw.max_iter = 3
    assig_fw.rgap_target = 1e-4
    assig_fw.execute()

    tc_fw = assig_fw.classes[0]
    assert tc_fw.results.total_turn_penalty > 0.0
