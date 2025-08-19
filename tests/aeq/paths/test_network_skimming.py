from os.path import join, isfile

import numpy as np

from aequilibrae.paths import skimming_single_origin
from aequilibrae.paths.multi_threaded_skimming import MultiThreadedNetworkSkimming
from aequilibrae.paths.network_skimming import NetworkSkimming
from aequilibrae.paths.results import SkimResults


def test_network_skimming(sioux_falls_example):
    network = sioux_falls_example.network
    project = sioux_falls_example
    proj_dir = sioux_falls_example.project_base_path

    network.build_graphs()
    graph = network.graphs["c"]
    graph.set_graph(cost_field="distance")
    graph.set_skimming("distance")
    graph.set_blocked_centroid_flows(False)

    # skimming results
    res = SkimResults()
    res.prepare(graph)
    aux_res = MultiThreadedNetworkSkimming()
    aux_res.prepare(graph, res.cores, res.nodes, res.num_skims)
    _ = skimming_single_origin(12, graph, res, aux_res, 0)

    skm = NetworkSkimming(graph)
    skm.execute()

    tot = np.nanmax(skm.results.skims.distance[:, :])
    assert tot <= np.sum(graph.cost), "Skimming was not successful. At least one np.inf returned."
    assert not skm.report, f"Skimming returned an error: {skm.report}"

    fn = "test_Skimming"
    skm.save_to_project(fn, format="omx")
    matrix_dir = join(proj_dir, "matrices")

    assert isfile(join(matrix_dir, f"{fn}.omx")), "Did not save project to project"

    matrices = project.matrices
    mat = matrices.get_record(fn)
    assert mat.name == fn, "Matrix record name saved wrong"
    assert mat.file_name == f"{fn}.omx", "matrix file_name saved wrong"
    assert mat.cores == 1, "matrix saved number of matrix cores wrong"
    assert mat.procedure == "Network skimming", "Matrix saved wrong procedure name"
    assert mat.procedure_id == skm.procedure_id, "Procedure ID saved wrong"
    assert mat.timestamp == skm.procedure_date, "Procedure ID saved wrong"
    project.close()


def test_network_skimming_no_project(sioux_falls_example):
    network = sioux_falls_example.network
    project = sioux_falls_example

    network.build_graphs()
    graph = network.graphs["c"]
    graph.set_graph(cost_field="distance")
    graph.set_skimming("distance")
    graph.set_blocked_centroid_flows(False)

    project.close()
    # skimming results
    res = SkimResults()
    res.prepare(graph)
    aux_res = MultiThreadedNetworkSkimming()
    aux_res.prepare(graph, res.cores, res.nodes, res.num_skims)
    _ = skimming_single_origin(12, graph, res, aux_res, 0)

    skm = NetworkSkimming(graph)
    skm.execute()

    tot = np.nanmax(skm.results.skims.distance[:, :])
    assert tot <= np.sum(graph.cost), "Skimming was not successful. At least one np.inf returned."
    assert not skm.report, f"Skimming returned an error: {skm.report}"
