import numpy as np
import pandas as pd

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph, TrafficAssignment, TrafficClass
from aequilibrae.project import Project


def test_standalone_graph_bidirectional_supernet_ids():
    df = pd.DataFrame(
        {
            "link_id": [10, 20, 30],
            "a_node": [1, 2, 3],
            "b_node": [2, 3, 4],
            "direction": [0, 1, -1],
            "distance": [100.0, 200.0, 300.0],
        }
    )

    g = Graph()
    g.network = df
    g.prepare_graph()

    assert len(g.graph) == 4
    supernet_ids = g.graph["__supernet_id__"].to_numpy()
    assert len(np.unique(supernet_ids)) == 4
    assert not g.graph["__supernet_id__"].duplicated().any()

    # Verify duplicate-healing fallback on corrupted input
    df_corrupted = df.copy()
    df_corrupted["__supernet_id__"] = np.array([0, 1, 2], dtype=np.int64)

    g_healed = Graph()
    g_healed.network = df_corrupted
    g_healed.prepare_graph()

    assert len(g_healed.graph) == 4
    assert len(np.unique(g_healed.graph["__supernet_id__"])) == 4
    assert not g_healed.graph["__supernet_id__"].duplicated().any()


def _build_test_multimodal_project(proj_path: str) -> Project:
    """Creates a 4-node multimodal project network with two-way and one-way links."""
    project = Project()
    project.new(proj_path)

    with project.db_connection as conn:
        conn.execute("INSERT OR IGNORE INTO modes (mode_id, mode_name) VALUES ('c', 'Car'), ('b', 'Bus');")
        for nid, is_c in [(1, 1), (2, 0), (3, 0), (4, 1)]:
            conn.execute(
                "INSERT INTO nodes (node_id, is_centroid, geometry) VALUES (?, ?, MakePoint(?, ?, 4326))",
                (nid, is_c, float(nid), float(nid)),
            )

        link_specs = [
            (1, 1, 2, 0, "cb", 10.0, 1000.0),
            (2, 2, 3, 0, "c", 15.0, 1000.0),
            (3, 3, 4, 0, "cb", 10.0, 1000.0),
            (4, 1, 4, 1, "b", 25.0, 500.0),
        ]
        sql_insert = (
            "INSERT INTO links (link_id, a_node, b_node, direction, modes, distance, link_type, "
            "travel_time_ab, travel_time_ba, capacity_ab, capacity_ba, geometry) "
            "VALUES (?, ?, ?, ?, ?, ?, 'default', ?, ?, ?, ?, MakeLine(MakePoint(?, ?, 4326), MakePoint(?, ?, 4326)))"
        )
        for lid, a, b, d, m, tt, cap in link_specs:
            conn.execute(
                sql_insert,
                (lid, a, b, d, m, tt, tt, tt, cap, cap, float(a), float(a), float(b), float(b)),
            )

    project.network.build_graphs()
    return project


def test_multimodal_project_bidirectional_supernet_parity(tmp_path):
    project = _build_test_multimodal_project(str(tmp_path / "test_proj"))

    gc = project.network.graphs["c"]
    gb = project.network.graphs["b"]

    assert len(gc.graph) == 6
    assert not gc.graph["__supernet_id__"].duplicated().any()
    assert len(np.unique(gc.graph["__supernet_id__"])) == 6

    assert len(gb.graph) == 5
    assert not gb.graph["__supernet_id__"].duplicated().any()
    assert len(np.unique(gb.graph["__supernet_id__"])) == 5

    common = pd.merge(
        gc.graph[["link_id", "direction", "__supernet_id__"]],
        gb.graph[["link_id", "direction", "__supernet_id__"]],
        on=["link_id", "direction"],
        suffixes=("_c", "_b"),
    )
    assert len(common) == 4
    assert (common["__supernet_id___c"] == common["__supernet_id___b"]).all()


def _build_demand_matrices(gc, gb):
    mat_c = AequilibraeMatrix()
    mat_c.create_empty(zones=gc.num_zones, matrix_names=["demand"], memory_only=True)
    mat_c.matrix["demand"][:, :] = 0.0
    mat_c.index[:] = gc.centroids[:]
    mat_c.matrix["demand"][0, 1] = 50.0
    mat_c.matrix["demand"][1, 0] = 30.0
    mat_c.computational_view(["demand"])

    mat_b = AequilibraeMatrix()
    mat_b.create_empty(zones=gb.num_zones, matrix_names=["demand"], memory_only=True)
    mat_b.matrix["demand"][:, :] = 0.0
    mat_b.index[:] = gb.centroids[:]
    mat_b.matrix["demand"][0, 1] = 20.0
    mat_b.computational_view(["demand"])
    return mat_c, mat_b


def test_multiclass_equilibrium_assignment_two_way_streets(tmp_path):
    project = _build_test_multimodal_project(str(tmp_path / "test_mc_proj"))
    gc = project.network.graphs["c"]
    gb = project.network.graphs["b"]
    gc.set_graph("travel_time")
    gb.set_graph("travel_time")

    mat_c, mat_b = _build_demand_matrices(gc, gb)
    tc = TrafficClass("car", gc, mat_c)
    tb = TrafficClass("bus", gb, mat_b)

    assig = TrafficAssignment()
    assig.set_classes([tc, tb])
    assig.set_vdf("BPR")
    assig.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assig.set_capacity_field("capacity")
    assig.set_time_field("travel_time")
    assig.set_algorithm("msa")
    assig.max_iter = 3

    preload_df = pd.DataFrame({"link_id": [1, 1], "direction": [1, -1], "preload": [5.0, 2.0]})
    assig.add_preload(preload_df)
    assig.execute()

    res = assig.results()
    assert not res.empty
    row_link1 = res.loc[1]
    assert row_link1["PCE_AB"] > 0
    assert row_link1["PCE_BA"] > 0
    assert row_link1["Preload_AB"] == 5.0
    assert row_link1["Preload_BA"] == 2.0


def test_multiclass_results_order_independent(tmp_path):
    """Verifies that assignment results include all links even when class 0 does not span the supernet."""
    project = _build_test_multimodal_project(str(tmp_path / "test_order_proj"))
    gc = project.network.graphs["c"]
    gb = project.network.graphs["b"]
    gc.set_graph("travel_time")
    gb.set_graph("travel_time")

    mat_c, mat_b = _build_demand_matrices(gc, gb)
    tc = TrafficClass("car", gc, mat_c)
    tb = TrafficClass("bus", gb, mat_b)

    # Put bus first (bus only serves links 1, 3, 4, missing car-only link 2)
    assig = TrafficAssignment()
    assig.set_classes([tb, tc])
    assig.set_vdf("BPR")
    assig.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assig.set_capacity_field("capacity")
    assig.set_time_field("travel_time")
    assig.set_algorithm("msa")
    assig.max_iter = 2
    assig.execute()

    res = assig.results()
    # All 4 project links must be present in the results table
    assert len(res) == 4
    assert 2 in res.index
    # Car-only link 2 must have positive PCE flow and valid congested time
    assert res.loc[2, "PCE_tot"] > 0
    assert res.loc[2, "Congested_Time_AB"] > 0


def test_coquimbo_traffic_assignment_execution(coquimbo_example):
    project = coquimbo_example
    project.network.build_graphs()
    gc = project.network.graphs["c"]

    assert len(gc.graph) == 34538
    assert not gc.graph["__supernet_id__"].duplicated().any()
    assert len(np.unique(gc.graph["__supernet_id__"])) == 34538

    gc.graph["capacity"] = 1000.0
    gc.graph["free_flow_time"] = 10.0
    gc.set_graph("free_flow_time")
    gc.cost = gc.graph["free_flow_time"].values

    mat = AequilibraeMatrix()
    mat.create_empty(zones=gc.num_zones, matrix_names=["demand"], memory_only=True)
    mat.matrix["demand"][:, :] = 0.1
    mat.index[:] = gc.centroids[:]
    mat.computational_view(["demand"])

    tc = TrafficClass("car", gc, mat)
    assig = TrafficAssignment()
    assig.set_classes([tc])
    assig.set_vdf("BPR")
    assig.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assig.set_capacity_field("capacity")
    assig.set_time_field("free_flow_time")
    assig.set_algorithm("msa")
    assig.max_iter = 1

    assig.execute()
    res = assig.results()
    assert len(res) == 19979
    assert "PCE_tot" in res.columns
    assert res["PCE_tot"].sum() > 0


def test_preload_is_indexed_by_supernet_id_not_graph_row(coquimbo_example):
    """Verifies a preload lands on the directed arc it names, in the supernet index space."""
    project = coquimbo_example
    project.network.build_graphs(modes=["c"])
    graph = project.network.graphs["c"]
    graph.set_graph("distance")
    graph.graph["capacity"] = 5000.0
    graph.graph["free_flow_time"] = graph.graph["distance"] / 1000.0

    centroids = np.array(graph.centroids, dtype=np.int64)
    mat = AequilibraeMatrix()
    mat.create_empty(zones=len(centroids), matrix_names=["demand"], memory_only=True)
    mat.index[:] = centroids[:]
    mat.computational_view(["demand"])
    mat.matrix_view[:, :] = 1.0

    assig = TrafficAssignment()
    assig.set_classes([TrafficClass("car", graph, mat)])
    assig.set_vdf("BPR")
    assig.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assig.set_capacity_field("capacity")
    assig.set_time_field("free_flow_time")
    assig.set_algorithm("msa")

    assert graph.num_links < graph.supernet_size

    rows = graph.graph[graph.graph.duplicated(subset="link_id", keep=False)]
    link_id = int(rows.link_id.iloc[0])
    both = graph.graph[graph.graph.link_id == link_id]
    ab_id = int(both.loc[both.direction == 1, "__supernet_id__"].iloc[0])
    ba_id = int(both.loc[both.direction == -1, "__supernet_id__"].iloc[0])

    assig.add_preload(pd.DataFrame({"link_id": [link_id], "direction": [-1], "preload": [777.0]}))

    assert len(assig.preloads) == graph.supernet_size
    vector = assig.assignment.preload
    assert vector.shape[0] == graph.supernet_size
    assert vector[ba_id] == 777.0
    assert vector[ab_id] == 0.0
    assert vector.sum() == 777.0
