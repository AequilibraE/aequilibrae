"""Test suite verifying turn-aware saved path-file comparison against independent oracle."""

from __future__ import annotations

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import tables

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph, TrafficAssignment, TrafficClass
from aequilibrae.paths.vdf import bpr


def _build_test_network():
    # 1 -> 2 -> 3: links 1 (cost 5), 2 (cost 5) = total cost 10
    # 1 -> 4 -> 3: links 3 (cost 7), 4 (cost 7) = total cost 14
    links = [
        {
            "link_id": 1,
            "a_node": 1,
            "b_node": 2,
            "direction": 1,
            "distance": 5.0,
            "free_flow_time": 5.0,
            "capacity": 1000.0,
        },
        {
            "link_id": 2,
            "a_node": 2,
            "b_node": 3,
            "direction": 1,
            "distance": 5.0,
            "free_flow_time": 5.0,
            "capacity": 1000.0,
        },
        {
            "link_id": 3,
            "a_node": 1,
            "b_node": 4,
            "direction": 1,
            "distance": 7.0,
            "free_flow_time": 7.0,
            "capacity": 1000.0,
        },
        {
            "link_id": 4,
            "a_node": 4,
            "b_node": 3,
            "direction": 1,
            "distance": 7.0,
            "free_flow_time": 7.0,
            "capacity": 1000.0,
        },
    ]
    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"
    return df


def _run_assignment_and_read_paths(
    net: pd.DataFrame,
    turns: pd.DataFrame | None,
    compress: bool,
) -> tuple[dict[tuple[int, int], list[int]], Graph]:
    centroids = np.array([1, 3], dtype=np.int64)
    g = Graph()
    g.network = net.copy()
    if turns is not None:
        g.set_turn_restrictions(turns)
    g.prepare_graph(centroids=centroids, remove_dead_ends=compress)
    g.set_graph("free_flow_time")

    mat = AequilibraeMatrix()
    mat.create_empty(memory_only=True, zones=2, matrix_names=["demand"])
    mat.index[:] = centroids
    mat.computational_view(core_list=["demand"])
    mat.matrix_view[:, :] = 0.0
    mat.matrix_view[0, 1] = 100.0  # 1 -> 3 demand

    with tempfile.TemporaryDirectory() as tmpdir:
        assig = TrafficAssignment()
        assig.set_classes([TrafficClass("car", g, mat)])
        assig.set_vdf(bpr, {"alpha": 0.15, "beta": 4.0})
        assig.set_capacity_field("capacity")
        assig.set_time_field("free_flow_time")
        assig.set_save_path_files(True)
        assig.set_algorithm("all-or-nothing")

        # Point assignment path output to tmpdir
        assig.assignment.project_path = tmpdir
        assig.execute()

        h5_path = Path(tmpdir) / "path_files.h5"
        assert h5_path.is_file(), f"No path file found in {list(Path(tmpdir).rglob('*'))}"

        with tables.open_file(h5_path, mode="r") as h5:
            grp = h5.root.iteration_1
            predecessors = grp.predecessors[:]
            connectors = grp.connectors[:]

    # The path file stores graph row indices; the tests speak in signed link IDs.
    signed_links = (g.graph.link_id.to_numpy(copy=False) * g.graph.direction.to_numpy(copy=False)).astype(np.int64)

    paths = {}
    for o_idx, o_val in enumerate(centroids):
        origin_idx = int(g.nodes_to_indices[o_val])
        for d_val in centroids:
            if d_val == o_val:
                continue
            node = int(g.nodes_to_indices[d_val])
            seq = []
            while node != origin_idx:
                conn = int(connectors[o_idx, node])
                pred = int(predecessors[o_idx, node])
                if conn < 0 or pred < 0:
                    seq = []
                    break
                seq.append(int(signed_links[conn]))
                node = pred
            if seq:
                paths[(int(o_val), int(d_val))] = list(reversed(seq))

    return paths, g


def test_saved_path_file_finite_penalty_modal_shift():
    """Verifies saved path file switches to detour when turn penalty exceeds detour cost differential."""
    net = _build_test_network()
    # Direct route cost: 10. Detour cost: 14.
    # Turn penalty 3.0: 10 + 3 = 13 < 14 -> stays on direct route [1, 2]
    turns_cheap = pd.DataFrame([{"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 3.0}])
    paths_cheap, _ = _run_assignment_and_read_paths(net, turns=turns_cheap, compress=False)
    assert paths_cheap[(1, 3)] == [1, 2]

    # Turn penalty 5.0: 10 + 5 = 15 > 14 -> switches to detour [3, 4]
    turns_expensive = pd.DataFrame([{"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 5.0}])
    paths_expensive, _ = _run_assignment_and_read_paths(net, turns=turns_expensive, compress=False)
    assert paths_expensive[(1, 3)] == [3, 4]


def test_saved_path_file_with_chain_compression():
    """Verifies that path files correctly unpack compressed chain links via mapping data."""
    net = _build_test_network()
    # With compress=True, 1 -> 4 -> 3 has node 4 with in=1, out=1, so it compresses into a single compact link
    # Turn restriction on 1 -> 2 -> 3 forces path through compressed chain 1 -> 4 -> 3
    prohib_turns = pd.DataFrame([{"from_node": 1, "via_node": 2, "to_node": 3, "penalty": np.inf}])
    paths_compressed, g = _run_assignment_and_read_paths(net, turns=prohib_turns, compress=True)

    assert (1, 3) in paths_compressed
    # 1 -> 4 -> 3 compressed into one compact arc, so the path file must unpack both original links
    assert paths_compressed[(1, 3)] == [3, 4]


def test_saved_path_file_signed_reverse_links():
    """Verifies that reverse (BA) link traversals unpack with signed negative link IDs without abs()."""
    # Link 1: 1 -> 2 (direction=1, cost=5)
    # Link 2: 3 -> 2 (direction=-1, meaning BA directed arc 2 -> 3, cost=5)
    # Link 3: 1 -> 3 (direction=1, cost=20)
    links = [
        {
            "link_id": 1,
            "a_node": 1,
            "b_node": 2,
            "direction": 1,
            "distance": 5.0,
            "free_flow_time": 5.0,
            "capacity": 1000.0,
        },
        {
            "link_id": 2,
            "a_node": 3,
            "b_node": 2,
            "direction": -1,
            "distance": 5.0,
            "free_flow_time": 5.0,
            "capacity": 1000.0,
        },
        {
            "link_id": 3,
            "a_node": 1,
            "b_node": 3,
            "direction": 1,
            "distance": 20.0,
            "free_flow_time": 20.0,
            "capacity": 1000.0,
        },
    ]
    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"

    # Route 1 -> 3 traverses link 1 (AB: +1) then link 2 in reverse (BA: -2)
    turns = pd.DataFrame([{"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 0.0}])
    paths, _ = _run_assignment_and_read_paths(df, turns=turns, compress=False)
    assert (1, 3) in paths
    assert paths[(1, 3)] == [1, -2]
