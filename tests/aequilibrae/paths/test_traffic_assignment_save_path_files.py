import os
import pathlib
import uuid
import zipfile
from os.path import join, dirname
from tempfile import gettempdir
from unittest import TestCase

import pandas as pd

from aequilibrae import TrafficAssignment, TrafficClass, Graph, Project
from ...data import siouxfalls_project


class TestTrafficAssignmentPathFiles(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

        proj_path = os.path.join(gettempdir(), "test_traffic_assignment_path_files" + uuid.uuid4().hex)
        os.mkdir(proj_path)
        zipfile.ZipFile(join(dirname(siouxfalls_project), "sioux_falls_single_class.zip")).extractall(proj_path)
        self.project = Project()
        self.project.open(proj_path)
        self.project.network.build_graphs()
        self.car_graph = self.project.network.graphs["c"]  # type: Graph
        self.car_graph.set_graph("free_flow_time")
        self.car_graph.set_blocked_centroid_flows(False)
        self.matrix = self.project.matrices.get_matrix("demand_omx")
        self.matrix.computational_view()

        self.assignment = TrafficAssignment()
        self.assigclass = TrafficClass("car", self.car_graph, self.matrix)

        self.algorithms = ["msa", "cfw", "bfw", "frank-wolfe"]

    def tearDown(self) -> None:
        self.matrix.close()
        self.project.close()

    def test_set_save_path_files(self):
        self.assignment.set_classes([self.assigclass])
        # make sure default is false
        for c in self.assignment.classes:
            self.assertEqual(c._aon_results.save_path_file, False)
        self.assignment.set_save_path_files(True)
        for c in self.assignment.classes:
            self.assertEqual(c._aon_results.save_path_file, True)

        # reset for most assignment tests
        self.assignment.set_save_path_files(False)
        for c in self.assignment.classes:
            self.assertEqual(c._aon_results.save_path_file, False)

    def test_set_path_file_format(self):
        self.assignment.set_classes([self.assigclass])
        with self.assertRaises(Exception):
            self.assignment.set_path_file_format("shiny_format")
        self.assignment.set_path_file_format("parquet")
        for c in self.assignment.classes:
            self.assertEqual(c._aon_results.write_feather, False)
        self.assignment.set_path_file_format("feather")
        for c in self.assignment.classes:
            self.assertEqual(c._aon_results.write_feather, True)

    def test_save_path_files(self):
        self.assignment.add_class(self.assigclass)
        self.assignment.set_save_path_files(True)

        self.assignment.set_vdf("BPR")
        self.assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        self.assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        self.assignment.set_capacity_field("capacity")
        self.assignment.set_time_field("free_flow_time")

        self.assignment.max_iter = 2
        self.assignment.set_algorithm("msa")
        self.assignment.execute()

        pid = self.assignment.procedure_id
        path_file_dir = pathlib.Path(join(str(self.project.project_base_path), "path_files", pid))
        self.assertTrue(path_file_dir.is_dir())

        # compare everything to reference files. Note that there is no graph simplification happening in SiouxFalls
        # and therefore we compare the files directly, otherwise a translation from the simplified ids to link_ids
        # would need to be performed.
        # Reference files were generated on 12/6/21, any changes to the test project will need to be applied to the
        # reference files. Also, the name given to the traffic class (see setUp above) has to be "car".
        class_id = f"c{self.assigclass.mode}_{self.assigclass.__id__}"
        reference_path_file_dir = pathlib.Path(siouxfalls_project) / "path_files"

        ref_node_correspondence = pd.read_feather(reference_path_file_dir / f"nodes_to_indices_{class_id}.feather")
        node_correspondence = pd.read_feather(path_file_dir / f"nodes_to_indices_{class_id}.feather")
        ref_node_correspondence.node_index = ref_node_correspondence.node_index.astype(
            node_correspondence.node_index.dtype
        )
        self.assertTrue(node_correspondence.equals(ref_node_correspondence))

        ref_correspondence = pd.read_feather(reference_path_file_dir / f"correspondence_{class_id}.feather")
        correspondence = pd.read_feather(path_file_dir / f"correspondence_{class_id}.feather")
        for col in correspondence.columns:
            ref_correspondence[col] = ref_correspondence[col].astype(correspondence[col].dtype)
        self.assertTrue(correspondence.equals(ref_correspondence))

        path_class_id = f"path_{class_id}"
        for i in range(1, self.assignment.max_iter + 1):
            class_dir = path_file_dir / f"iter{i}" / path_class_id
            ref_class_dir = reference_path_file_dir / f"iter{i}" / path_class_id
            for o in self.assigclass.matrix.index:
                o_ind = self.assigclass.graph.compact_nodes_to_indices[o]
                this_o_path_file = pd.read_feather(class_dir / f"o{o_ind}.feather")
                ref_this_o_path_file = pd.read_feather(ref_class_dir / f"o{o_ind}.feather")
                pd.testing.assert_frame_equal(ref_this_o_path_file, this_o_path_file)

                this_o_index_file = pd.read_feather(class_dir / f"o{o_ind}_indexdata.feather")
                ref_this_o_index_file = pd.read_feather(ref_class_dir / f"o{o_ind}_indexdata.feather")
                pd.testing.assert_frame_equal(ref_this_o_index_file, this_o_index_file)
