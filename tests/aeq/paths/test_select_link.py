import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from aequilibrae import TrafficAssignment, TrafficClass, Graph, PathResults
from aequilibrae.matrix import AequilibraeMatrix


@pytest.fixture
def select_link_setup(sioux_falls_single_class):
    sioux_falls_single_class.network.build_graphs()
    car_graph = sioux_falls_single_class.network.graphs["c"]  # type: Graph
    car_graph.set_graph("free_flow_time")
    car_graph.set_blocked_centroid_flows(False)
    matrix = sioux_falls_single_class.matrices.get_matrix("demand_omx")
    matrix.computational_view()

    assignment = TrafficAssignment()
    assignclass = TrafficClass("car", car_graph, matrix)
    assignment.set_classes([assignclass])
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("free_flow_time")
    assignment.max_iter = 1
    assignment.set_algorithm("msa")

    yield {
        "project": sioux_falls_single_class,
        "car_graph": car_graph,
        "matrix": matrix,
        "assignment": assignment,
        "assignclass": assignclass,
    }

    matrix.close()


def test_multiple_link_sets(select_link_setup):
    """
    Tests whether the Select Link feature works as wanted.
    Uses two examples: 2 links in one select link, and a single Selected Link
    Checks both the OD Matrix and Link Loading
    """
    assignclass = select_link_setup["assignclass"]
    assignment = select_link_setup["assignment"]
    project = select_link_setup["project"]

    assignclass.set_select_links({"sl_9_or_6": [(9, 1), (6, 1)], "just_3": [(3, 1)], "sl_5_for_fun": [(5, 1)]})
    assignment.execute()

    for key in assignclass._selected_links.keys():
        od_mask, link_loading = create_od_mask(
            assignclass.matrix.matrix_view, assignclass.graph, assignclass._selected_links[key]
        )
        np.testing.assert_allclose(
            assignclass.results.select_link_od.matrix[key][:, :, 0],
            od_mask,
            err_msg=f"OD SL matrix for: {key} does not match",
        )
        np.testing.assert_allclose(
            assignclass.results.select_link_loading[key],
            link_loading,
            err_msg=f"Link loading SL matrix for: {key} does not match",
        )

    # Test if files are saved in the right place
    assignment.save_select_link_results("select_link_analysis")

    matrices = project.matrices
    matrices.update_database()
    assert "select_link_analysis.omx" in matrices.list()["file_name"].tolist()

    # Test if matrices are with the correct shape and are not empty
    sla = matrices.get_matrix("select_link_analysis_omx")
    num_zones = assignment.classes[0].graph.num_zones
    for mat in sla.names:
        m = sla.get_matrix(mat)
        assert m.sum() > 0 and m.shape == (num_zones, num_zones)

    results = project.results.list()["table_name"].to_list()
    assert "select_link_analysis" in results


def test_equals_demand_one_origin(select_link_setup):
    """
    Test to ensure the Select Link functionality behaves as required.
    Tests to make sure the OD matrix works when all links surrounding one origin are selected
    Confirms the Link Loading is done correctly in this case
    """
    assignclass = select_link_setup["assignclass"]
    assignment = select_link_setup["assignment"]

    assignclass.set_select_links({"sl_1_4_3_and_2": [(1, 1), (4, 1), (3, 1), (2, 1)]})
    assignment.execute()

    for key in assignclass._selected_links.keys():
        od_mask, link_loading = create_od_mask(
            assignclass.matrix.matrix_view, assignclass.graph, assignclass._selected_links[key]
        )
        np.testing.assert_allclose(
            assignclass.results.select_link_od.matrix[key][:, :, 0],
            od_mask,
            err_msg=f"OD SL matrix for: {key} does not match",
        )
        np.testing.assert_allclose(
            assignclass.results.select_link_loading[key],
            link_loading,
            err_msg=f"Link loading SL matrix for: {key} does not match",
        )


def test_single_demand(select_link_setup):
    """
    Tests the functionality of Select Link when given a custom demand matrix, where only 1 OD pair has demand on it
    Confirms the OD matrix behaves, and the Link Loading is just on the path of this OD pair
    """
    assignclass = select_link_setup["assignclass"]
    assignment = select_link_setup["assignment"]
    matrix = select_link_setup["matrix"]

    custom_demand = np.zeros((24, 24, 1)).astype(float)
    custom_demand[0, 23, 0] = 1000
    matrix.matrix_view = custom_demand
    assignclass.matrix = matrix

    assignclass.set_select_links({"sl_39_66_or_73": [(39, 1), (66, 1), (73, 1)]})
    assignment.execute()

    for key in assignclass._selected_links.keys():
        od_mask, link_loading = create_od_mask(
            assignclass.matrix.matrix_view, assignclass.graph, assignclass._selected_links[key]
        )
        np.testing.assert_allclose(
            assignclass.results.select_link_od.matrix[key][:, :, 0],
            od_mask,
            err_msg=f"OD SL matrix for: {key} does not match",
        )
        np.testing.assert_allclose(
            assignclass.results.select_link_loading[key],
            link_loading,
            err_msg=f"Link loading SL matrix for: {key} does not match",
        )


def test_select_link_network_loading(select_link_setup):
    """
    Test to ensure the SL_network_loading method correctly does the network loading
    """
    assignclass = select_link_setup["assignclass"]
    assignment = select_link_setup["assignment"]
    car_graph = select_link_setup["car_graph"]
    matrix = select_link_setup["matrix"]

    # First run without select links
    assignment.execute()
    non_sl_loads = assignclass.results.get_load_results()

    # Create new setup for select links
    new_assignment = TrafficAssignment()
    new_assignclass = TrafficClass("car", car_graph, matrix)
    new_assignment.set_classes([new_assignclass])
    new_assignment.set_vdf("BPR")
    new_assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    new_assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
    new_assignment.set_capacity_field("capacity")
    new_assignment.set_time_field("free_flow_time")
    new_assignment.max_iter = 1
    new_assignment.set_algorithm("msa")

    new_assignclass.set_select_links({"sl_39_66_or_73": [(39, 1), (66, 1), (73, 1)]})
    new_assignment.execute()
    sl_loads = new_assignclass.results.get_load_results()

    np.testing.assert_allclose(non_sl_loads.matrix_tot, sl_loads.matrix_tot)


def test_duplicate_links(select_link_setup):
    """
    Tests to make sure the user api correctly filters out duplicate links in the compressed graph
    """
    car_graph = select_link_setup["car_graph"]
    matrix = select_link_setup["matrix"]

    assignclass = TrafficClass("car", car_graph, matrix)

    with pytest.warns(Warning):
        assignclass.set_select_links({"test": [(1, 1), (1, 1)]})

    assert len(assignclass._selected_links["test"]) == 1, "Did not correctly remove duplicate link"


def test_link_out_of_bounds(select_link_setup):
    """
    Test to confirm the user api correctly identifies when an input node is invalid for the current graph
    """
    car_graph = select_link_setup["car_graph"]
    matrix = select_link_setup["matrix"]

    assignclass = TrafficClass("car", car_graph, matrix)

    with pytest.raises(ValueError):
        assignclass.set_select_links({"test": [(78, 1), (1, 1)]})


def test_kaitang(test_data_path, tmp_path):
    zipfile.ZipFile(test_data_path / "KaiTang.zip").extractall(tmp_path)

    link_df = pd.read_csv(os.path.join(tmp_path, "link.csv"))
    centroids_array = np.array([7, 8, 11])

    net = link_df.copy()

    g = Graph()
    g.network = net
    g.network_ok = True
    g.status = "OK"
    g.mode = "a"
    g.prepare_graph(centroids_array)
    g.set_blocked_centroid_flows(False)
    g.set_graph("fft")

    aem_mat = AequilibraeMatrix()
    aem_mat.load(os.path.join(tmp_path, "demand_a.aem"))
    aem_mat.computational_view(["a"])

    assign_class = TrafficClass("class_a", g, aem_mat)
    assign_class.set_fixed_cost("a_toll")
    assign_class.set_vot(1.1)
    assign_class.set_select_links(links={"trace": [(7, 0), (13, 0)]})

    assign = TrafficAssignment()
    assign.set_classes([assign_class])
    assign.set_vdf("BPR")
    assign.set_vdf_parameters({"alpha": "alpha", "beta": "beta"})
    assign.set_capacity_field("capacity")
    assign.set_time_field("fft")
    assign.set_algorithm("bfw")
    assign.max_iter = 100
    assign.rgap_target = 0.0001

    # 4.execute
    assign.execute()

    # 5.receive results
    assign_flow_res_df = assign.results().sort_index().reset_index(drop=False).astype(float).fillna(0.0)
    select_link_flow_df = assign.select_link_flows().sort_index().reset_index(drop=False).astype(float).fillna(0.0)

    pd.testing.assert_frame_equal(
        assign_flow_res_df[["link_id", "a_ab", "a_ba", "a_tot"]],
        select_link_flow_df.rename(
            columns={"class_a_trace_a_ab": "a_ab", "class_a_trace_a_ba": "a_ba", "class_a_trace_a_tot": "a_tot"}
        )[["link_id", "a_ab", "a_ba", "a_tot"]],
    )


@pytest.mark.parametrize("algorithm", ["all-or-nothing", "msa", "fw", "cfw", "bfw"])
def test_multi_iteration(select_link_setup, algorithm):
    car_graph = select_link_setup["car_graph"]
    matrix = select_link_setup["matrix"]

    assignment = TrafficAssignment()
    assignclass = TrafficClass("car", car_graph, matrix)
    assignment.set_classes([assignclass])
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("free_flow_time")
    assignment.max_iter = 10
    assignment.set_algorithm(algorithm)

    assignclass.set_select_links({"sl_1_1": [(1, 1)], "sl_5_1": [(5, 1)]})
    assignment.execute()

    assignment_results = assignclass.results.get_load_results()
    sl_results = assignclass.results.get_sl_results()

    assert abs(assignment_results["matrix_ab"].loc[1] - sl_results["sl_1_1_matrix_ab"].loc[1]) < 1e-6, (
        f"Select link results differ to that of the assignment ({algorithm})"
    )

    assert abs(assignment_results["matrix_ab"].loc[5] - sl_results["sl_5_1_matrix_ab"].loc[5]) < 1e-6, (
        f"Select link results differ to that of the assignment ({algorithm})"
    )


def create_od_mask(demand: np.array, graph: Graph, sl):
    res = PathResults()
    # This uses the UNCOMPRESSED graph, since we don't know which nodes the user may ask for
    graph.set_graph("free_flow_time")
    res.prepare(graph)

    def g(o, d):
        res.compute_path(o, d)
        return list(res.path_nodes) if (res.path_nodes is not None and o != d) else []

    a = [[g(o, d) for d in range(1, 25)] for o in range(1, 25)]
    sl_links = []
    for i in range(len(sl)):
        node_pair = graph.graph.iloc[sl[i]]["a_node"] + 1, graph.graph.iloc[sl[i]]["b_node"] + 1
        sl_links.append(node_pair)
    mask = {}
    for origin, val in enumerate(a):
        for dest, path in enumerate(val):
            for k in range(1, len(path)):
                if origin == dest:
                    pass
                elif (path[k - 1], path[k]) in sl_links:
                    mask[(origin, dest)] = True
    sl_od = np.zeros((24, 24))
    for origin in range(24):
        for dest in range(24):
            if mask.get((origin, dest)):
                sl_od[origin, dest] = demand[origin, dest][0]

    # make link loading
    loading = np.zeros((76, 1))
    for orig, dest in mask.keys():
        path = a[orig][dest]
        for i in range(len(path) - 1):
            link = (
                graph.graph[(graph.graph["a_node"] == path[i] - 1) & (graph.graph["b_node"] == path[i + 1] - 1)][
                    "link_id"
                ].values[0]
                - 1
            )
            loading[link] += demand[orig, dest]
    return sl_od, loading
