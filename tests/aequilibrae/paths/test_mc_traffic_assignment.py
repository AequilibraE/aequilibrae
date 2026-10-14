import pytest

from aequilibrae import TrafficAssignment, TrafficClass
from aequilibrae.paths.vdf import bpr


@pytest.fixture
def assignment_setup(sioux_falls_example):
    classes, matrices = [], []
    for name, mode, core, pce in (
        ("car", "c", "car", 1.0),
        ("truck", "T", "trucks", 2.5),
        ("motorcycle", "M", "motorcycle", 0.2),
    ):
        matrix = sioux_falls_example.matrices.get_matrix("demand_mc", subset=[core])
        matrices.append(matrix)
        classes.append(
            TrafficClass(
                name=name,
                mode=mode,
                cost="free_flow_time",
                demand=matrix,
                pce=pce,
                skimming=["free_flow_time", "distance"],
                block_centroid_flows=False,
            )
        )
    yield TrafficAssignment(
        network=sioux_falls_example.network.build(),
        classes=classes,
        time_field="free_flow_time",
        capacity_field="capacity",
        vdf=bpr,
        name_mapping={"alpha": "b", "beta": "power"},
        algorithm="bfw",
        project=sioux_falls_example,
    )
    for matrix in matrices:
        matrix.close()


def test_execute_and_save_results(assignment_setup):
    assignment = assignment_setup
    assignment.execute(max_iterations=20, cores=1)
    assert set(assignment.class_results) == {"car", "truck", "motorcycle"}
    assignment.save_results("save_to_database")
    assignment.save_skims(matrix_name="my_skims", which_ones="all")
    with pytest.raises(ValueError):
        assignment.save_results("save_to_database")
    with pytest.raises(FileExistsError):
        assignment.save_skims(matrix_name="my_skims", which_ones="all")
