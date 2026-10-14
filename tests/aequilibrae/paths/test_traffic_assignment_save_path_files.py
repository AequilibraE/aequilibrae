"""Unsupported path output is not part of the traffic assignment API."""

import pytest

from aequilibrae import TrafficAssignment, TrafficClass
from aequilibrae.paths import available_heaps
from aequilibrae.paths.vdf import bpr


@pytest.mark.parametrize("heap", available_heaps())
def test_assignment_accepts_available_heaps(sioux_falls_single_class, heap):
    project = sioux_falls_single_class
    matrix = project.matrices.get_matrix("demand_omx")
    matrix.computational_view()
    traffic = TrafficClass(name="car", mode="c", cost="free_flow_time", demand=matrix, heap=heap)
    assignment = TrafficAssignment(
        network=project.network.build(modes=["c"]),
        classes=[traffic],
        time_field="free_flow_time",
        capacity_field="capacity",
        vdf=bpr,
        algorithm="all-or-nothing",
    )
    assignment.execute(cores=1)
    assert assignment.class_results["car"]._heap == heap
    assert not hasattr(assignment, "set_save_path_files")
    with pytest.raises(TypeError):
        assignment.execute(save_path_files=True)
    matrix.close()
