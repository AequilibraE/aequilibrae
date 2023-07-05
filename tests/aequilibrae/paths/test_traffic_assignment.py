import random
import sqlite3
import string
from os.path import join, isfile
from pathlib import Path
from random import choice

import numpy as np
import pandas as pd
import pytest

from aequilibrae import TrafficAssignment, TrafficClass, Graph, Project
from aequilibrae.utils.create_example import create_example
from ...data import siouxfalls_project


@pytest.fixture
def project(tmp_path):
    proj = create_example(str(tmp_path / "test_traffic_assignment"))
    proj.network.build_graphs()
    proj.activate()
    return proj


@pytest.fixture
def car_graph(project):
    graph: Graph = project.network.graphs["c"]
    graph.set_graph("free_flow_time")
    graph.set_blocked_centroid_flows(False)
    return graph


@pytest.fixture
def matrix(project):
    mat = project.matrices.get_matrix("demand_omx")
    mat.computational_view()
    return mat


@pytest.fixture
def assigclass(car_graph, matrix):
    return TrafficClass("car", car_graph, matrix)


@pytest.fixture
def assignment(project):
    return TrafficAssignment(project)


class TestTrafficAssignmentSetup:
    algorithms = ["msa", "cfw", "bfw", "frank-wolfe"]

    def test_matrix_with_wrong_type(self, matrix, car_graph):
        matrix.matrix_view = np.array(matrix.matrix_view, np.int32)
        with pytest.raises(TypeError):
            TrafficClass("car", car_graph, matrix)

    def test_set_vdf(self, assignment: TrafficAssignment):
        with pytest.raises(ValueError):
            assignment.set_vdf("CQS")
        assignment.set_vdf("BPR")

    def test_set_classes(self, assignment: TrafficAssignment, assigclass: TrafficClass):
        with pytest.raises(AttributeError):
            assignment.set_classes([1, 2])

        with pytest.raises(TypeError):
            assignment.set_classes(assigclass)

        assignment.set_classes([assigclass])

    def test_algorithms_available(self, assignment: TrafficAssignment):
        algs = assignment.algorithms_available()
        real = ["all-or-nothing", "msa", "frank-wolfe", "bfw", "cfw", "fw"]

        diff = [x for x in real if x not in algs]
        diff2 = [x for x in algs if x not in real]

        assert len(diff) + len(diff2) <= 0, "list of algorithms raised is wrong"

    def test_set_cores(self, assignment: TrafficAssignment, assigclass: TrafficClass):
        with pytest.raises(Exception):
            assignment.set_cores(3)

        assignment.add_class(assigclass)
        with pytest.raises(ValueError):
            assignment.set_cores("q")

        assignment.set_cores(3)

    def test_set_algorithm(self, assignment: TrafficAssignment, assigclass: TrafficClass):
        with pytest.raises(AttributeError):
            assignment.set_algorithm("not an algo")

        assignment.add_class(assigclass)

        with pytest.raises(Exception):
            assignment.set_algorithm("msa")

        assignment.set_vdf("BPR")
        assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        assignment.set_capacity_field("capacity")
        assignment.set_time_field("free_flow_time")

        assignment.max_iter = 10

        for algo in self.algorithms:
            for _ in range(10):
                algo = "".join([x.upper() if random.random() < 0.5 else x.lower() for x in algo])
                assignment.set_algorithm(algo)

        with pytest.raises(AttributeError):
            assignment.set_algorithm("not a valid algorithm")

    def test_set_vdf_parameters(self, assignment: TrafficAssignment, assigclass: TrafficClass):
        with pytest.raises(RuntimeError):
            assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        assignment.set_vdf("bpr")
        assignment.add_class(assigclass)
        assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

    def test_set_time_field(self, assignment: TrafficAssignment, assigclass: TrafficClass):
        with pytest.raises(ValueError):
            assignment.set_time_field("capacity")

        assignment.add_class(assigclass)

        N = random.randint(1, 50)
        val = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
        with pytest.raises(ValueError):
            assignment.set_time_field(val)

        assignment.set_time_field("free_flow_time")
        assert assignment.time_field == "free_flow_time"

    def test_set_capacity_field(self, assignment: TrafficAssignment, assigclass: TrafficClass):
        with pytest.raises(ValueError):
            assignment.set_capacity_field("capacity")

        assignment.add_class(assigclass)

        N = random.randint(1, 50)
        val = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
        with pytest.raises(ValueError):
            assignment.set_capacity_field(val)

        assignment.set_capacity_field("capacity")
        assert assignment.capacity_field == "capacity"

    def test_info(self, assignment: TrafficAssignment, assigclass: TrafficClass):
        iterations = random.randint(1, 10000)
        rgap = random.random() / 10000
        algo = choice(self.algorithms)

        assignment.add_class(assigclass)
        assignment.set_vdf("BPR")
        assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        assignment.set_capacity_field("capacity")
        assignment.set_time_field("free_flow_time")

        assignment.max_iter = iterations
        assignment.rgap_target = rgap
        assignment.set_algorithm(algo)

        # TY
        for _ in range(10):
            algo = "".join([x.upper() if random.random() < 0.5 else x.lower() for x in algo])

        dct = assignment.info()
        if algo.lower() == "fw":
            algo = "frank-wolfe"
        assert dct["Algorithm"] == algo.lower(), "Algorithm not correct in info method"

        assert dct["Maximum iterations"] == iterations, "maximum iterations not correct in info method"


class TestTrafficAssignment:
    @pytest.fixture(params=["memmap", "memonly"])
    def matrix(self, request, matrix):
        if request.param == "memonly":
            return matrix.copy(memory_only=True)
        return matrix

    def test_execute_and_save_results(
        self, assignment: TrafficAssignment, assigclass: TrafficClass, car_graph: Graph, matrix
    ):
        conn = sqlite3.connect(join(siouxfalls_project, "project_database.sqlite"))
        results = pd.read_sql("select volume from links order by link_id", conn)

        proj = assignment.project
        assignment.add_class(assigclass)
        assignment.set_vdf("BPR")
        assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        assignment.set_capacity_field("capacity")
        assignment.set_time_field("free_flow_time")

        assignment.max_iter = 10
        assignment.set_algorithm("msa")
        assignment.execute()

        msa10_rgap = assignment.assignment.rgap

        correl = np.corrcoef(assigclass.results.total_link_loads, results.volume.values)[0, 1]
        assert 0.8 < correl

        assignment.max_iter = 500
        assignment.rgap_target = 0.001
        assignment.set_algorithm("msa")
        assignment.execute()
        msa25_rgap = assignment.assignment.rgap

        correl = np.corrcoef(assigclass.results.total_link_loads, results.volume)[0, 1]
        assert 0.98 < correl

        assignment.set_algorithm("frank-wolfe")
        assignment.execute()

        fw25_rgap = assignment.assignment.rgap
        fw25_iters = assignment.assignment.iter

        correl = np.corrcoef(assigclass.results.total_link_loads, results.volume)[0, 1]
        assert 0.99 < correl

        assignment.set_algorithm("cfw")
        assignment.execute()
        cfw25_rgap = assignment.assignment.rgap
        cfw25_iters = assignment.assignment.iter

        correl = np.corrcoef(assigclass.results.total_link_loads, results.volume)[0, 1]
        assert 0.995 < correl

        # For the last algorithm, we set skimming
        car_graph.set_skimming(["free_flow_time", "distance"])
        assigclass = TrafficClass("car", car_graph, matrix)
        assignment.set_classes([assigclass])

        assignment.set_algorithm("bfw")
        assignment.execute()
        bfw25_rgap = assignment.assignment.rgap
        bfw25_iters = assignment.assignment.iter

        correl = np.corrcoef(assigclass.results.total_link_loads, results.volume)[0, 1]
        assert 0.999 < correl

        assert msa25_rgap < msa10_rgap
        # MSA and FW do not reach 1e-4 within 500 iterations, cfw and bfw do
        assert fw25_rgap < msa25_rgap
        assert cfw25_rgap < assignment.rgap_target
        assert bfw25_rgap < assignment.rgap_target
        # we expect bfw to converge quicker than cfw
        assert cfw25_iters < fw25_iters
        assert bfw25_iters < cfw25_iters

        assignment.save_results("save_to_database")
        assignment.save_skims(matrix_name="all_skims", which_ones="all")

        with pytest.raises(ValueError):
            assignment.save_results("save_to_database")

        num_cores = assignment.cores
        # Let's test logging of assignment
        log_ = Path(proj.path_to_file).parent / "aequilibrae.log"
        assert isfile(log_)

        file_text = ""
        with open(log_, "r", encoding="utf-8") as file:
            for line in file.readlines():
                file_text += line

        tc_spec = "INFO ; Traffic Class specification"
        assert file_text.count(tc_spec) > 1

        tc_graph = "INFO ; {'car': {'Graph': \"{'Mode': 'c', 'Block through centroids': False, 'Number of centroids': 24, 'Links': 76, 'Nodes': 24}\","
        assert file_text.count(tc_graph) > 1

        tc_matrix = "'Number of centroids': 24, 'Matrix cores': ['matrix'], 'Matrix totals': {'matrix': 360600.0}}\"}}"
        assert file_text.count(tc_matrix) > 1

        assig_1 = "INFO ; {{'VDF parameters': {{'alpha': 'b', 'beta': 'power'}}, 'VDF function': 'bpr', 'Number of cores': {}, 'Capacity field': 'capacity', 'Time field': 'free_flow_time', 'Algorithm': 'msa', 'Maximum iterations': 10, 'Target RGAP': 0.0001}}".format(
            num_cores
        )
        assert assig_1 in file_text

        assig_2 = "INFO ; {{'VDF parameters': {{'alpha': 'b', 'beta': 'power'}}, 'VDF function': 'bpr', 'Number of cores': {}, 'Capacity field': 'capacity', 'Time field': 'free_flow_time', 'Algorithm': 'msa', 'Maximum iterations': 500, 'Target RGAP': 0.001}}".format(
            num_cores
        )
        assert assig_2 in file_text

        assig_3 = "INFO ; {{'VDF parameters': {{'alpha': 'b', 'beta': 'power'}}, 'VDF function': 'bpr', 'Number of cores': {}, 'Capacity field': 'capacity', 'Time field': 'free_flow_time', 'Algorithm': 'frank-wolfe', 'Maximum iterations': 500, 'Target RGAP': 0.001}}".format(
            num_cores
        )
        assert assig_3 in file_text

        assig_4 = "INFO ; {{'VDF parameters': {{'alpha': 'b', 'beta': 'power'}}, 'VDF function': 'bpr', 'Number of cores': {}, 'Capacity field': 'capacity', 'Time field': 'free_flow_time', 'Algorithm': 'cfw', 'Maximum iterations': 500, 'Target RGAP': 0.001}}".format(
            num_cores
        )
        assert assig_4 in file_text

        assig_5 = "INFO ; {{'VDF parameters': {{'alpha': 'b', 'beta': 'power'}}, 'VDF function': 'bpr', 'Number of cores': {}, 'Capacity field': 'capacity', 'Time field': 'free_flow_time', 'Algorithm': 'bfw', 'Maximum iterations': 500, 'Target RGAP': 0.001}}".format(
            num_cores
        )
        assert assig_5 in file_text

    def test_execute_no_project(self, project: Project, assignment: TrafficAssignment, assigclass: TrafficClass):
        conn = sqlite3.connect(join(siouxfalls_project, "project_database.sqlite"))
        results = pd.read_sql("select volume from links order by link_id", conn)

        project.close()

        assignment = TrafficAssignment()
        assignment.add_class(assigclass)
        assignment.set_vdf("BPR")
        assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

        assignment.set_capacity_field("capacity")
        assignment.set_time_field("free_flow_time")

        assignment.max_iter = 10
        assignment.set_algorithm("msa")
        assignment.execute()

        correl = np.corrcoef(assigclass.results.total_link_loads, results.volume.values)[0, 1]
        assert 0.8 < correl

        with pytest.raises(FileNotFoundError):
            assignment.save_results("anything")
