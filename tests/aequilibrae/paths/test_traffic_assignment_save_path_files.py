"""Unsupported assignment path output must not be silently ignored."""

import pytest

from aequilibrae import TrafficAssignment, TrafficClass


@pytest.fixture
def assignment(sioux_falls_single_class):
    project = sioux_falls_single_class
    project.network.build_graphs()
    graph = project.network.graphs["c"]
    graph.set_graph("free_flow_time")
    matrix = project.matrices.get_matrix("demand_omx")
    matrix.computational_view()
    result = TrafficAssignment()
    result.set_classes([TrafficClass("car", graph, matrix)])
    yield result
    matrix.close()


def test_path_saving_is_explicitly_unsupported(assignment):
    assignment.set_save_path_files(False)
    with pytest.raises(NotImplementedError, match="Path file saving"):
        assignment.set_save_path_files(True)
    assert not assignment.classes[0]._aon_results.save_path_file


@pytest.mark.parametrize("format", ["feather", "parquet"])
def test_path_file_format_is_explicitly_unsupported(assignment, format):
    with pytest.raises(NotImplementedError, match="path file formats"):
        assignment.set_path_file_format(format)


@pytest.mark.parametrize("heap", ["std", "pairing"])
def test_assignment_heap_selection_is_explicitly_unsupported(assignment, heap):
    with pytest.raises(NotImplementedError, match="4ary"):
        assignment.classes[0].set_heap(heap)
