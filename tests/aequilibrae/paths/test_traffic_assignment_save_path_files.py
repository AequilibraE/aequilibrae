import numpy as np
import pytest
import tables

from aequilibrae import TrafficAssignment, TrafficClass


@pytest.fixture
def assignment_setup(sioux_falls_single_class):
    project = sioux_falls_single_class
    project.network.build_graphs()
    car_graph = project.network.graphs["c"]
    car_graph.set_graph("free_flow_time")
    car_graph.set_blocked_centroid_flows(False)
    matrix = project.matrices.get_matrix("demand_omx")
    matrix.computational_view()
    assignment = TrafficAssignment()
    assigclass = TrafficClass("car", car_graph, matrix)
    algorithms = ["msa", "cfw", "bfw", "frank-wolfe"]
    yield {
        "project": project,
        "car_graph": car_graph,
        "matrix": matrix,
        "assignment": assignment,
        "assigclass": assigclass,
        "algorithms": algorithms,
    }
    matrix.close()


def test_set_save_path_files(assignment_setup):
    assignment = assignment_setup["assignment"]
    assigclass = assignment_setup["assigclass"]
    assignment.set_classes([assigclass])
    # make sure default is false
    for c in assignment.classes:
        assert c._aon_results.save_path_file is False
    assignment.set_save_path_files(True)
    for c in assignment.classes:
        assert c._aon_results.save_path_file is True
    # reset for most assignment tests
    assignment.set_save_path_files(False)
    for c in assignment.classes:
        assert c._aon_results.save_path_file is False


def test_save_path_files(assignment_setup):
    """Verify HDF5 path files produce valid paths whose link loads match a pure AoN assignment."""
    assignment = assignment_setup["assignment"]
    assigclass = assignment_setup["assigclass"]
    project = assignment_setup["project"]
    graph = assignment_setup["car_graph"]
    matrix = assignment_setup["matrix"]

    # Single-iteration AoN with path file saving
    assignment.add_class(assigclass)
    assignment.set_save_path_files(True)
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("free_flow_time")
    assignment.max_iter = 1
    assignment.set_algorithm("msa")
    assignment.execute()

    h5_path = project.project_base_path / "path_files.h5"
    assert h5_path.is_file(), f"Expected HDF5 file at {h5_path}"

    a_nodes = graph.graph["a_node"].to_numpy()
    b_nodes = graph.graph["b_node"].to_numpy()

    num_origins = len(assigclass.matrix.index)
    num_network_nodes = len(graph.all_nodes)
    origin_ids = assigclass.matrix.index

    # Reference link loads from the AoN assignment (indexed by graph row)
    ref_link_loads = assignment.results()["PCE_tot"].to_numpy()

    # --- Reconstruct link loads from the HDF5 paths + demand matrix ---
    with tables.open_file(h5_path, mode="r") as h5:
        grp = h5.root.iteration_1
        assert h5.get_node_attr(grp, "iteration") == 1
        assert h5.get_node_attr(grp, "num_origins") == num_origins
        assert h5.get_node_attr(grp, "num_network_nodes") == num_network_nodes

        preds = grp.predecessors
        conns = grp.connectors
        assert preds.shape == (num_origins, num_network_nodes)
        assert conns.shape == (num_origins, num_network_nodes)

        reconstructed = np.zeros(graph.num_links, dtype=np.float64)

        for o_idx, origin in enumerate(origin_ids):
            origin_node_idx = graph.nodes_to_indices[origin]
            predecessors = preds[o_idx, :]
            connectors = conns[o_idx, :]

            dest_mask = matrix.matrix_view[o_idx, :, 0] > 0
            dest_nodes = np.where(dest_mask)[0]

            for dest_compact_idx in dest_nodes:
                demand = matrix.matrix_view[o_idx, dest_compact_idx, 0]
                path = _trace_path(predecessors, connectors, dest_compact_idx, origin_node_idx)
                assert len(path) > 0, f"Empty path from {origin} to dest idx {dest_compact_idx}"
                _assert_valid_path(path, a_nodes, b_nodes, origin_node_idx, dest_compact_idx)

                for conn in path:
                    reconstructed[conn] += demand

    # Reconstructed link loads must match the AoN assignment exactly
    np.testing.assert_array_equal(reconstructed, ref_link_loads)


def _trace_path(predecessors, connectors, dest, origin_node_idx):
    """Walk predecessors back from *dest* to *origin_node_idx*, returning the
    sequence of connector (graph-index / __supernet_id__) values along the path."""
    path = []
    cur = dest
    while cur != origin_node_idx:
        conn = connectors[cur]
        if conn < 0:
            break
        path.append(conn)
        cur = predecessors[cur]
        if len(path) > 1000:
            break
    return list(reversed(path))


def _assert_valid_path(path, a_nodes, b_nodes, origin_node_idx, dest_node_idx):
    """Verify that *path* (a list of graph indices / __supernet_id__) forms a
    continuous chain from *origin_node_idx* to *dest_node_idx*."""
    assert len(path) >= 1, "Path must have at least one link"

    # First link should start at the origin
    first_conn = path[0]
    assert a_nodes[first_conn] == origin_node_idx, (
        f"First link a_node={a_nodes[first_conn]} != origin={origin_node_idx}"
    )

    # Walk the chain: each link's b_node must equal the next link's a_node
    cur_node = b_nodes[first_conn]
    for conn in path[1:]:
        assert a_nodes[conn] == cur_node, f"Link {conn}: a_node={a_nodes[conn]} != expected a_node={cur_node}"
        cur_node = b_nodes[conn]

    # Last link should end at the destination
    assert cur_node == dest_node_idx, f"Final b_node={cur_node} != destination={dest_node_idx}"
