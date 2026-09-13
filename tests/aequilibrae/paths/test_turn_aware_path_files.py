"""Test suite verifying turn-aware saved path-file comparison against independent oracle."""

from __future__ import annotations

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph, TrafficAssignment, TrafficClass


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
    use_hybrid: bool,
) -> tuple[dict[tuple[int, int], list[int]], Graph]:
    centroids = np.array([1, 3], dtype=np.int64)
    g = Graph()
    g.network = net.copy()
    if turns is not None:
        g.set_turn_restrictions(turns)
    g.prepare_graph(centroids=centroids, remove_dead_ends=compress)
    g.set_graph("free_flow_time")
    g.set_hybrid_kernel(use_hybrid)

    mat = AequilibraeMatrix()
    mat.create_empty(memory_only=True, zones=2, matrix_names=["demand"])
    mat.index[:] = centroids
    mat.computational_view(core_list=["demand"])
    mat.matrix_view[:, :] = 0.0
    mat.matrix_view[0, 1] = 100.0  # 1 -> 3 demand

    with tempfile.TemporaryDirectory() as tmpdir:
        assig = TrafficAssignment()
        assig.set_classes([TrafficClass("car", g, mat)])
        assig.set_vdf("BPR")
        assig.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        assig.set_capacity_field("capacity")
        assig.set_time_field("free_flow_time")
        assig.set_save_path_files(True)
        assig.set_path_file_format("parquet")
        assig.set_algorithm("all-or-nothing")

        # Point assignment path output to tmpdir
        assig.assignment.project_path = Path(tmpdir)
        assig.execute()

        path_files = list(Path(tmpdir).rglob("o0.parquet"))
        assert len(path_files) > 0, f"No path files found in {list(Path(tmpdir).rglob('*'))}"
        path_dir = path_files[0].parent

        paths = {}
        for o_idx, o_val in enumerate(centroids):
            path_file = path_dir / f"o{o_idx}.parquet"
            idx_file = path_dir / f"o{o_idx}_indexdata.parquet"
            if not path_file.exists() or not idx_file.exists():
                continue
            df_path = pd.read_parquet(path_file)
            df_idx = pd.read_parquet(idx_file)

            path_data = df_path["data"].to_numpy()
            idx_data = df_idx["data"].to_numpy()

            for d_idx, d_val in enumerate(centroids):
                if o_idx == d_idx:
                    continue
                start = 0 if d_idx == 0 else int(idx_data[d_idx - 1])
                end = int(idx_data[d_idx])
                if start < end:
                    # Stored in destination-to-origin order, reverse to origin-to-destination
                    seq = [int(abs(x)) for x in reversed(path_data[start:end])]
                    paths[(int(o_val), int(d_val))] = seq

    return paths, g


def test_saved_path_file_unrestricted_vs_prohibited():
    """Verifies saved path file switches from direct route to detour when turn is prohibited."""
    net = _build_test_network()

    # 1. Unrestricted (zero penalty): shortest path 1 -> 3 is 1 -> 2 -> 3 (links 1, 2)
    zero_turns = pd.DataFrame([{"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 0.0}])
    paths_unrestricted, _ = _run_assignment_and_read_paths(net, turns=zero_turns, compress=False, use_hybrid=True)
    assert (1, 3) in paths_unrestricted
    assert paths_unrestricted[(1, 3)] == [1, 2]

    # 2. Prohibited turn 1 -> 2 -> 3: shortest path must switch to detour 1 -> 4 -> 3 (links 3, 4)
    prohib_turns = pd.DataFrame([{"from_node": 1, "via_node": 2, "to_node": 3, "penalty": np.inf}])
    paths_prohibited, _ = _run_assignment_and_read_paths(net, turns=prohib_turns, compress=False, use_hybrid=True)
    assert (1, 3) in paths_prohibited
    assert paths_prohibited[(1, 3)] == [3, 4]


def test_saved_path_file_finite_penalty_modal_shift():
    """Verifies saved path file switches to detour when turn penalty exceeds detour cost differential."""
    net = _build_test_network()
    # Direct route cost: 10. Detour cost: 14.
    # Turn penalty 3.0: 10 + 3 = 13 < 14 -> stays on direct route [1, 2]
    turns_cheap = pd.DataFrame([{"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 3.0}])
    paths_cheap, _ = _run_assignment_and_read_paths(net, turns=turns_cheap, compress=False, use_hybrid=True)
    assert paths_cheap[(1, 3)] == [1, 2]

    # Turn penalty 5.0: 10 + 5 = 15 > 14 -> switches to detour [3, 4]
    turns_expensive = pd.DataFrame([{"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 5.0}])
    paths_expensive, _ = _run_assignment_and_read_paths(net, turns=turns_expensive, compress=False, use_hybrid=True)
    assert paths_expensive[(1, 3)] == [3, 4]


def test_saved_path_file_with_chain_compression():
    """Verifies that path files correctly unpack compressed chain links via mapping data."""
    net = _build_test_network()
    # With compress=True, 1 -> 4 -> 3 has node 4 with in=1, out=1, so it compresses into a single compact link
    # Turn restriction on 1 -> 2 -> 3 forces path through compressed chain 1 -> 4 -> 3
    prohib_turns = pd.DataFrame([{"from_node": 1, "via_node": 2, "to_node": 3, "penalty": np.inf}])
    paths_compressed, g = _run_assignment_and_read_paths(net, turns=prohib_turns, compress=True, use_hybrid=True)

    assert (1, 3) in paths_compressed
    # 1 -> 4 -> 3 compressed into one compact arc, so the path file must unpack both original links
    assert paths_compressed[(1, 3)] == [3, 4]


def test_saved_path_file_hybrid_vs_arc_parity():
    """Verifies that saved path files produced by hybrid and arc-based kernels are identical."""
    net = _build_test_network()
    turns = pd.DataFrame([{"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 20.0}])

    paths_hybrid, _ = _run_assignment_and_read_paths(net, turns=turns, compress=True, use_hybrid=True)
    paths_arc, _ = _run_assignment_and_read_paths(net, turns=turns, compress=True, use_hybrid=False)

    assert paths_hybrid == paths_arc
    assert paths_hybrid[(1, 3)] == [3, 4]
