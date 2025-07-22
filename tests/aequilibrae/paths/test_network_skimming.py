import os
import uuid
from os.path import join, isfile
from shutil import rmtree
from tempfile import gettempdir
import pytest

import numpy as np

from aequilibrae.paths import skimming_single_origin
from aequilibrae.paths.multi_threaded_skimming import MultiThreadedNetworkSkimming
from aequilibrae.paths.network_skimming import NetworkSkimming
from aequilibrae.paths.results import SkimResults
from aequilibrae.utils.create_example import create_example


@pytest.fixture
def network_setup():
    os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

    proj_dir = os.path.join(gettempdir(), uuid.uuid4().hex)
    project = create_example(proj_dir)
    network = project.network

    yield {"proj_dir": proj_dir, "project": project, "network": network}

    # Teardown
    try:
        rmtree(proj_dir)
    except Exception as e:
        print(f"Failed to remove at {e.args}")


def test_network_skimming(network_setup):
    network = network_setup["network"]
    project = network_setup["project"]
    proj_dir = network_setup["proj_dir"]

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


def test_network_skimming_no_project(network_setup):
    network = network_setup["network"]
    project = network_setup["project"]

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
