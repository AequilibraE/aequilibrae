"""Unsupported assignment path output must not be silently ignored."""

from dataclasses import replace

import pytest

from aequilibrae import TrafficAssignment, TrafficClass
from aequilibrae.paths import available_heaps
from aequilibrae.paths.vdf import bpr


@pytest.fixture
def assignment(sioux_falls_single_class):
    project = sioux_falls_single_class
    matrix = project.matrices.get_matrix("demand_omx", subset=["matrix"])
    result = TrafficAssignment(
        network=project.network,
        classes=[TrafficClass(name="car", mode="c", cost="free_flow_time", demand=matrix)],
        time_field="free_flow_time",
        capacity_field="capacity",
        vdf=bpr,
        algorithm="all-or-nothing",
    )
    yield result


def test_path_saving_is_explicitly_unsupported(assignment):
    assert not hasattr(assignment, "set_save_path_files")
    with pytest.raises(TypeError):
        assignment.execute(save_path_files=True)


@pytest.mark.parametrize("format", ["feather", "parquet"])
def test_path_file_format_is_explicitly_unsupported(assignment, format):
    assert not hasattr(assignment, "set_path_file_format")
    with pytest.raises(TypeError):
        assignment.execute(path_file_format=format)


@pytest.mark.parametrize("heap", available_heaps())
def test_assignment_accepts_available_heaps(assignment, heap):
    traffic = replace(assignment.classes[0], heap=heap)
    assignment = TrafficAssignment(
        network=assignment.network,
        classes=[traffic],
        time_field="free_flow_time",
        capacity_field="capacity",
        vdf=bpr,
        algorithm="all-or-nothing",
    )
    assignment.execute(cores=1)
    assert assignment.class_results["car"]._heap == heap
    assert assignment._class_states[0]._aon_results._heap == heap
