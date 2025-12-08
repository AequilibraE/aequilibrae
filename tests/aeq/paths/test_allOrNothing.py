import pytest

from aequilibrae.paths import Graph
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae.paths.results import AssignmentResults


@pytest.fixture(scope="function")
def project(sioux_falls_example):
    sioux_falls_example.network.build_graphs()
    yield sioux_falls_example


def build_graph(project):
    project.network.build_graphs()
    g = project.network.graphs["c"]  # type: Graph
    g.set_graph("distance")
    g.set_skimming("distance")
    g.set_blocked_centroid_flows(False)
    return g


def matrix_aem(project):
    mat = project.matrices.get_matrix("demand_aem")
    mat.computational_view()
    return mat


def matrix_omx(project):
    matrix = project.matrices.get_matrix("demand_omx")
    matrix.computational_view()
    matrix.matrix_view *= 2
    return matrix


def test_skimming_on_assignment(sioux_falls_example):
    matrix = matrix_aem(sioux_falls_example)
    graph = build_graph(sioux_falls_example)
    res = AssignmentResults()
    res.prepare(graph, matrix)

    graph.set_skimming([])
    graph.set_blocked_centroid_flows(True)
    assig = allOrNothing("name", matrix, graph, res)
    assig.execute()

    assert res.skims.distance.sum() == 0, (
        "skimming for nothing during assignment returned something different than zero"
    )

    res.prepare(graph, matrix)
    assig = allOrNothing("name", matrix, graph, res)
    assig.execute()


def test_execute(sioux_falls_example):
    matrix = matrix_aem(sioux_falls_example)
    matrix2 = matrix_omx(sioux_falls_example)
    graph = build_graph(sioux_falls_example)
    # Loads and prepares the graph
    res1 = AssignmentResults()
    res1.prepare(graph, matrix)
    assig1 = allOrNothing("name", matrix, graph, res1)
    assig1.execute()

    res2 = AssignmentResults()
    res2.prepare(graph, matrix2)
    assig2 = allOrNothing("name", matrix2, graph, res2)
    assig2.execute()

    load1 = res1.get_load_results()
    load2 = res2.get_load_results()

    assert list(load1.matrix_tot * 2) == list(load2.matrix_tot), "Something wrong with the AoN"
    matrix2.close()
