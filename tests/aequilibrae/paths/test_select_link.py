import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from aequilibrae import TrafficAssignment, TrafficClass, NetworkData, Graph, PathResults
from aequilibrae.paths.vdf import bpr
from aequilibrae.matrix import AequilibraeMatrix, from_file


@pytest.fixture
def select_link_setup(sioux_falls_single_class):
    project = sioux_falls_single_class
    network = project.network.build(modes=["c"])
    matrix = project.matrices.get_matrix("demand_omx")

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

    yield {"project": project, "matrix": matrix, "make": make}
    matrix.close()


@pytest.mark.parametrize(
    "links, single_demand",
    [
        ({"sl_9_or_6": [(9, 1), (6, 1)], "just_3": [(3, 1)], "sl_5_for_fun": [(5, 1)]}, False),
        ({"sl_1_4_3_and_2": [(1, 1), (4, 1), (3, 1), (2, 1)]}, False),
        ({"sl_39_66_or_73": [(39, 1), (66, 1), (73, 1)]}, True),
    ],
)
def test_selected_demand_and_loads_match_paths(select_link_setup, links, single_demand):
    matrix = select_link_setup["matrix"]
    if single_demand:
        matrix["matrix"] = np.zeros((24, 24), dtype=float)
        matrix["matrix"][0, 23] = 1000
    assignment = select_link_setup["make"](links)
    assignment.execute(max_iterations=1, cores=1)
    state = assignment._class_states[0]
    for key in state._selected_links:
        od_mask, link_loading = create_od_mask(matrix, state.graph, state._selected_links[key])
        np.testing.assert_allclose(state.results.select_link_od.matrices[key][:, :, 0], od_mask)
        np.testing.assert_allclose(state.results.select_link_loading[key], link_loading)
    assignment.save_select_link_results("select_link_analysis")
    project = select_link_setup["project"]
    project.matrices.update_database()
    assert "select_link_analysis.omx" in project.matrices.list()["file_name"].tolist()
    sla = project.matrices.get_matrix("select_link_analysis_omx")
    for name in sla:
        assert sla[name].sum() > 0
        assert sla[name].shape == (24, 24)
    assert "select_link_analysis" in project.results.list()["table_name"].tolist()
    sla.close()


def test_select_link_analysis_does_not_change_loading(select_link_setup):
    plain = select_link_setup["make"]()
    plain.execute(max_iterations=1, cores=1)
    selected = select_link_setup["make"]({"selected": [(39, 1), (66, 1), (73, 1)]})
    selected.execute(max_iterations=1, cores=1)
    np.testing.assert_allclose(plain.class_results["car"].link_loads, selected.class_results["car"].link_loads)


def test_duplicate_selected_links_are_removed(select_link_setup):
    assignment = select_link_setup["make"]({"test": [(1, 1), (1, 1)]})
    assignment.execute(max_iterations=1, cores=1)
    assert len(assignment._class_states[0]._selected_links["test"]) == 1


def test_invalid_selected_link_fails_on_construction(select_link_setup):
    with pytest.raises(ValueError, match="Selected link"):
        select_link_setup["make"]({"test": [(78, 1), (1, 1)]})


def test_kaitang(test_data_path, tmp_path):
    zipfile.ZipFile(test_data_path / "KaiTang.zip").extractall(tmp_path)

    link_df = pd.read_csv(os.path.join(tmp_path, "link.csv"))
    centroids_array = np.array([7, 8, 11])

    net = link_df.copy()

    net["modes"] = "a"
    aem_mat = from_file(os.path.join(tmp_path, "demand_a.aem"))
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
        network=NetworkData(net, centroids=centroids_array),
        classes=[traffic],
        vdf=bpr,
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


@pytest.mark.parametrize("algorithm", ["all-or-nothing", "msa", "frank-wolfe", "cfw", "bfw"])
def test_multi_iteration(select_link_setup, algorithm):
    assignment = select_link_setup["make"]({"sl_1_1": [(1, 1)], "sl_5_1": [(5, 1)]}, algorithm)
    assignment.execute(max_iterations=10, cores=1)
    results = assignment.class_results["car"]
    loads, selected = results.get_load_results(), results.get_sl_results()
    for link in (1, 5):
        assert abs(loads["matrix_ab"].loc[link] - selected[f"sl_{link}_1_matrix_ab"].loc[link]) < 1e-6


def create_od_mask(demand: AequilibraeMatrix, graph: Graph, sl):
    demand = np.stack([demand[name] for name in demand], axis=2)
    # This uses the UNCOMPRESSED graph, since we don't know which nodes the user may ask for
    graph.set_graph("free_flow_time")

    def g(o, d):
        res = PathResults(graph, o, d)
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
