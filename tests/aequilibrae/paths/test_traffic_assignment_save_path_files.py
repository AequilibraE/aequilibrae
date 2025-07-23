import pandas as pd
import pytest

from aequilibrae import TrafficAssignment, TrafficClass, Graph, Project


@pytest.fixture
def assignment_setup(sioux_falls_single_class):
    project = sioux_falls_single_class
    project.network.build_graphs()
    car_graph = project.network.graphs["c"]
    car_graph.set_graph("free_flow_time")
    car_graph.set_blocked_centroid_flows(False)
    matrix = project.matrices.get_matrix("demand_omx")
    matrix.computational_view()
    assignment = TrafficAssignment()
    assigclass = TrafficClass("car", car_graph, matrix)
    algorithms = ["msa", "cfw", "bfw", "frank-wolfe"]
    yield {
        "project": project,
        "car_graph": car_graph,
        "matrix": matrix,
        "assignment": assignment,
        "assigclass": assigclass,
        "algorithms": algorithms,
    }
    matrix.close()
    project.close()


def test_set_save_path_files(assignment_setup):
    assignment = assignment_setup["assignment"]
    assigclass = assignment_setup["assigclass"]
    assignment.set_classes([assigclass])
    # make sure default is false
    for c in assignment.classes:
        assert c._aon_results.save_path_file is False
    assignment.set_save_path_files(True)
    for c in assignment.classes:
        assert c._aon_results.save_path_file is True
    # reset for most assignment tests
    assignment.set_save_path_files(False)
    for c in assignment.classes:
        assert c._aon_results.save_path_file is False


def test_set_path_file_format(assignment_setup):
    assignment = assignment_setup["assignment"]
    assigclass = assignment_setup["assigclass"]
    assignment.set_classes([assigclass])
    with pytest.raises(Exception):
        assignment.set_path_file_format("shiny_format")
    assignment.set_path_file_format("parquet")
    for c in assignment.classes:
        assert c._aon_results.write_feather is False
    assignment.set_path_file_format("feather")
    for c in assignment.classes:
        assert c._aon_results.write_feather is True


def test_save_path_files(assignment_setup, sioux_falls_test):
    assignment = assignment_setup["assignment"]
    assigclass = assignment_setup["assigclass"]
    project = assignment_setup["project"]
    assignment.add_class(assigclass)
    assignment.set_save_path_files(True)
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("free_flow_time")
    assignment.max_iter = 2
    assignment.set_algorithm("msa")
    assignment.execute()
    pid = assignment.procedure_id
    path_file_dir = project.project_base_path / "path_files" / pid
    assert path_file_dir.is_dir()

    class_id = f"c{assigclass.mode}_{assigclass._id}"
    reference_path_file_dir = sioux_falls_test.project_base_path / "path_files"
    ref_node_correspondence = pd.read_feather(reference_path_file_dir / f"nodes_to_indices_{class_id}.feather")
    node_correspondence = pd.read_feather(path_file_dir / f"nodes_to_indices_{class_id}.feather")
    ref_node_correspondence.node_index = ref_node_correspondence.node_index.astype(node_correspondence.node_index.dtype)
    assert node_correspondence.equals(ref_node_correspondence)

    ref_correspondence = pd.read_feather(reference_path_file_dir / f"correspondence_{class_id}.feather")
    correspondence = pd.read_feather(path_file_dir / f"correspondence_{class_id}.feather")
    for col in correspondence.columns:
        ref_correspondence[col] = ref_correspondence[col].astype(correspondence[col].dtype)
    assert correspondence.equals(ref_correspondence)

    path_class_id = f"path_{class_id}"
    for i in range(1, assignment.max_iter + 1):
        class_dir = path_file_dir / f"iter{i}" / path_class_id
        ref_class_dir = reference_path_file_dir / f"iter{i}" / path_class_id
        for o in assigclass.matrix.index:
            o_ind = assigclass.graph.compact_nodes_to_indices[o]
            this_o_path_file = pd.read_feather(class_dir / f"o{o_ind}.feather")
            ref_this_o_path_file = pd.read_feather(ref_class_dir / f"o{o_ind}.feather")
            pd.testing.assert_frame_equal(ref_this_o_path_file, this_o_path_file)
            this_o_index_file = pd.read_feather(class_dir / f"o{o_ind}_indexdata.feather")
            ref_this_o_index_file = pd.read_feather(ref_class_dir / f"o{o_ind}_indexdata.feather")
            pd.testing.assert_frame_equal(ref_this_o_index_file, this_o_index_file)
