import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from aequilibrae import TrafficAssignment, TrafficClass, Network, Graph, PathResults
from aequilibrae.paths.vdf import bpr
from aequilibrae.matrix import AequilibraeMatrix, from_file


@pytest.fixture
def select_link_setup(sioux_falls_single_class):
    project = sioux_falls_single_class
    network = project.network
    matrix = project.matrices.get_matrix("demand_omx", subset=["matrix"])

    def make(links=None, algorithm="msa"):
        traffic = TrafficClass(
            name="car",
            mode="c",
            cost="free_flow_time",
            demand=matrix,
            block_centroid_flows=False,
            select_link_sets=links or {},
        )
        return TrafficAssignment(
            network=network,
            classes=[traffic],
            vdf=bpr,
            time_field="free_flow_time",
            capacity_field="capacity",
            algorithm=algorithm,
            name_mapping={"alpha": "b", "beta": "power"},
            project=project,
        )

    yield {
        "project": project,
        "car_graph": network.build_graph("c", "free_flow_time", block_centroid_flows=False),
        "matrix": matrix,
        "make": make,
    }


def test_multiple_link_sets(select_link_setup):
    """
    Tests whether the Select Link feature works as wanted.
    Uses two examples: 2 links in one select link, and a single Selected Link
    Checks both the OD Matrix and Link Loading
    """
    project = select_link_setup["project"]

    assignment = select_link_setup["make"](
        {"sl_9_or_6": [(9, 1), (6, 1)], "just_3": [(3, 1)], "sl_5_for_fun": [(5, 1)]}
    )
    assignment.execute(max_iterations=1, cores=1)

    for key in assignment.classes[0].select_link_sets:
        od_mask, link_loading = create_od_mask(
            select_link_setup["matrix"], select_link_setup["car_graph"], assignment.classes[0].select_link_sets[key]
        )
        np.testing.assert_allclose(assignment.class_results["car"].select_link_od.matrices[key][:, :, 0], od_mask)
        np.testing.assert_allclose(assignment.class_results["car"].select_link_loading[key], link_loading)

    # Test if files are saved in the right place
    assignment.save_select_link_results("select_link_analysis")

    matrices = project.matrices
    matrices.update_database()
    assert "select_link_analysis.omx" in matrices.list()["file_name"].tolist()

    # Test if matrices are with the correct shape and are not empty
    sla = matrices.get_matrix("select_link_analysis_omx")
    num_zones = len(assignment.classes[0].demand.index)
    for mat in sla:
        m = sla[mat]
        assert m.sum() > 0 and m.shape == (num_zones, num_zones)

    sla.close()

    results = project.results.list()["table_name"].to_list()
    assert "select_link_analysis" in results


def test_equals_demand_one_origin(select_link_setup):
    """
    Test to ensure the Select Link functionality behaves as required.
    Tests to make sure the OD matrix works when all links surrounding one origin are selected
    Confirms the Link Loading is done correctly in this case
    """

    assignment = select_link_setup["make"]({"sl_1_4_3_and_2": [(1, 1), (4, 1), (3, 1), (2, 1)]})
    assignment.execute(max_iterations=1, cores=1)

    for key in assignment.classes[0].select_link_sets:
        od_mask, link_loading = create_od_mask(
            select_link_setup["matrix"], select_link_setup["car_graph"], assignment.classes[0].select_link_sets[key]
        )
        np.testing.assert_allclose(assignment.class_results["car"].select_link_od.matrices[key][:, :, 0], od_mask)
        np.testing.assert_allclose(assignment.class_results["car"].select_link_loading[key], link_loading)


def test_single_demand(select_link_setup):
    """
    Tests the functionality of Select Link when given a custom demand matrix, where only 1 OD pair has demand on it
    Confirms the OD matrix behaves, and the Link Loading is just on the path of this OD pair
    """
    matrix = select_link_setup["matrix"]

    custom_demand = np.zeros((24, 24, 1)).astype(float)
    custom_demand[0, 23, 0] = 1000
    core = next(iter(matrix))
    matrix[core] = custom_demand[:, :, 0]

    assignment = select_link_setup["make"]({"sl_39_66_or_73": [(39, 1), (66, 1), (73, 1)]})
    assignment.execute(max_iterations=1, cores=1)

    for key in assignment.classes[0].select_link_sets:
        od_mask, link_loading = create_od_mask(
            select_link_setup["matrix"], select_link_setup["car_graph"], assignment.classes[0].select_link_sets[key]
        )
        np.testing.assert_allclose(assignment.class_results["car"].select_link_od.matrices[key][:, :, 0], od_mask)
        np.testing.assert_allclose(assignment.class_results["car"].select_link_loading[key], link_loading)


def test_select_link_network_loading(select_link_setup):
    """
    Test to ensure the SL_network_loading method correctly does the network loading
    """
    # First run without select links
    assignment = select_link_setup["make"]()
    assignment.execute(max_iterations=1, cores=1)
    non_sl_loads = assignment.class_results["car"].get_load_results()

    # Create new setup for select links
    new_assignment = select_link_setup["make"]({"sl_39_66_or_73": [(39, 1), (66, 1), (73, 1)]})
    new_assignment.execute(max_iterations=1, cores=1)
    sl_loads = new_assignment.class_results["car"].get_load_results()

    np.testing.assert_allclose(non_sl_loads.matrix_tot, sl_loads.matrix_tot)


def test_duplicate_links(select_link_setup):
    """
    Tests to make sure the user api correctly filters out duplicate links in the compressed graph
    """
    assignment = select_link_setup["make"]({"test": [(1, 1), (1, 1)]})
    assignment.execute(max_iterations=1, cores=1)
    assert len(assignment._class_states[0]._selected_links["test"]) == 1, "Did not correctly remove duplicate link"


def test_link_out_of_bounds(select_link_setup):
    """
    Test to confirm the user api correctly identifies when an input node is invalid for the current graph
    """
    with pytest.raises(ValueError, match="Selected link"):
        select_link_setup["make"]({"test": [(78, 1), (1, 1)]}).execute(cores=1)


def test_kaitang(test_data_path, tmp_path):
    zipfile.ZipFile(test_data_path / "KaiTang.zip").extractall(tmp_path)

    link_df = pd.read_csv(os.path.join(tmp_path, "link.csv"))
    centroids_array = np.array([7, 8, 11])

    net = link_df.copy()

    net["modes"] = "a"
    aem_mat = AequilibraeMatrix(index=centroids_array)
    aem_mat["a"] = np.ones((len(centroids_array), len(centroids_array)))
    traffic = TrafficClass(
        name="class_a",
        mode="a",
        cost="fft",
        demand=aem_mat,
        fixed_cost_field="a_toll",
        vot=1.1,
        block_centroid_flows=False,
        select_link_sets={"trace": [(7, 0), (13, 0)]},
    )
    assign = TrafficAssignment(
        network=Network(net, centroids=centroids_array),
        classes=[traffic],
        vdf=bpr,
        name_mapping={"alpha": "alpha", "beta": "beta"},
        time_field="fft",
        capacity_field="capacity",
        algorithm="bfw",
    )
    assign.execute(max_iterations=100, rgap=0.0001, cores=1)

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
    assignment = select_link_setup["make"]({"sl_1_1": [(1, 1)], "sl_5_1": [(5, 1)]}, algorithm)
    assignment.execute(max_iterations=10, cores=1)
    results = assignment.class_results["car"]
    assignment_results = results.get_load_results()
    sl_results = results.get_sl_results()

    assert abs(assignment_results["matrix_ab"].loc[1] - sl_results["sl_1_1_matrix_ab"].loc[1]) < 1e-6, (
        f"Select link results differ to that of the assignment ({algorithm})"
    )

    assert abs(assignment_results["matrix_ab"].loc[5] - sl_results["sl_5_1_matrix_ab"].loc[5]) < 1e-6, (
        f"Select link results differ to that of the assignment ({algorithm})"
    )


def create_od_mask(demand: AequilibraeMatrix, graph: Graph, sl):
    demand = np.stack([demand[name] for name in demand], axis=2)
    # This uses the UNCOMPRESSED graph, since we don't know which nodes the user may ask for

    def g(o, d):
        res = PathResults(graph, o, d)
        return list(res.path_nodes) if (res.path_nodes is not None and o != d) else []

    a = [[g(o, d) for d in range(1, 25)] for o in range(1, 25)]
    sl_links = []
    for link_id, direction in sl:
        mask = graph.full_link_ids == link_id
        if direction:
            mask &= graph.full_directions == direction
        sl_links.extend(zip(
            graph.full_node_ids[graph.full_tails[mask]], graph.full_node_ids[graph.full_heads[mask]], strict=True
        ))
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
            mask = (graph.full_node_ids[graph.full_tails] == path[i]) & (
                graph.full_node_ids[graph.full_heads] == path[i + 1]
            )
            link = graph.full_link_ids[mask][0] - 1
            loading[link] += demand[orig, dest]
    return sl_od, loading


def test_select_link_export_aligns_different_centroid_sets(sioux_falls_single_class):
    project = sioux_falls_single_class
    network = Network(pd.DataFrame({
        "link_id": [11, 12, 13], "a_node": [1, 2, 3], "b_node": [2, 3, 4],
        "direction": [0, 0, 0], "modes": ["c"] * 3,
        "time": [1.0] * 3, "capacity": [100.0] * 3,
    }))
    classes = []
    for name, centroids in (("first", [3, 1, 2]), ("second", [4, 2])):
        demand = AequilibraeMatrix(index=np.array(centroids))
        demand["flow"] = np.ones((len(centroids), len(centroids)), dtype=np.float64)
        classes.append(TrafficClass(
            name=name, mode="c", cost="time", demand=demand, block_centroid_flows=False,
            select_link_sets={"chosen": [11, 12, 13]},
        ))
    assignment = TrafficAssignment(
        network=network, classes=classes, time_field="time", capacity_field="capacity",
        vdf=bpr, name_mapping={"alpha": 0.0, "beta": 1.0}, algorithm="msa", project=project,
    )
    assignment.execute(max_iterations=1, cores=1)
    assignment.save_select_link_matrices("different_centroids", project=project)
    exported = from_file(project.matrices.folder / "different_centroids.omx")
    np.testing.assert_array_equal(exported.index, [1, 2, 3, 4])
    for traffic in classes:
        source = assignment.class_results[traffic.name].select_link_od.matrices["chosen"][:, :, 0]
        assert source.sum() > 0
        expected = np.zeros((4, 4))
        positions = np.searchsorted(exported.index, traffic.demand.index)
        expected[np.ix_(positions, positions)] = source
        np.testing.assert_array_equal(exported[f"chosen_{traffic.name}_flow"], expected)
