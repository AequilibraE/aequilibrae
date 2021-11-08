from unittest import TestCase
import numpy as np
from aequilibrae.paths.network_skimming import NetworkSkimming
from aequilibrae.paths import skimming_single_origin
from aequilibrae.paths.results import SkimResults
from aequilibrae.paths.multi_threaded_skimming import MultiThreadedNetworkSkimming
import os
from shutil import copytree, rmtree
import uuid
from tempfile import gettempdir
from aequilibrae.utils.create_example import create_example


class TestNetwork_skimming(TestCase):
    def setUp(self) -> None:
        os.environ['PATH'] = os.path.join(gettempdir(), 'temp_data') + ';' + os.environ['PATH']

        self.proj_dir = os.path.join(gettempdir(), uuid.uuid4().hex)

        self.project = create_example(self.proj_dir)
        self.network = self.project.network
        self.curr = self.project.conn.cursor()

    def tearDown(self) -> None:
        self.project.close()
        del self.curr
        try:
            rmtree(self.proj_dir)
        except Exception as e:
            print(f'Failed to remove at {e.args}')

    def test_network_skimming(self):

        self.network.build_graphs()
        graph = self.network.graphs['c']
        graph.set_graph(cost_field="distance")
        graph.set_skimming("distance")
        graph.set_blocked_centroid_flows(False)

        # skimming results
        res = SkimResults()
        res.prepare(graph)
        aux_res = MultiThreadedNetworkSkimming()
        aux_res.prepare(graph, res)
        _ = skimming_single_origin(12, graph, res, aux_res, 0)

        skm = NetworkSkimming(graph)
        skm.execute()

        tot = np.nanmax(skm.results.skims.distance[:, :])
        if tot > np.sum(graph.cost):
            self.fail("Skimming was not successful. At least one np.inf returned.")

        if skm.report:
            self.fail("Skimming returned an error:" + str(skm.report))

        skm.save_to_project('tEst_Skimming')

        if not os.path.isfile(os.path.join(self.proj_dir, 'matrices', 'test_skimming.omx')):
            self.fail('Did not save project to project')

        matrices = self.project.matrices
        mat = matrices.get_record('TEsT_Skimming')
        self.assertEqual(mat.name, 'tEst_Skimming', 'Matrix record name saved wrong')
        self.assertEqual(mat.file_name, 'tEst_Skimming.omx', 'matrix file_name saved  wrong')
        self.assertEqual(mat.cores, 1, 'matrix saved number of matrix cores wrong')
        self.assertEqual(mat.procedure, 'Network skimming', 'Matrix saved wrong procedure name')
        self.assertEqual(mat.procedure_id, skm.procedure_id, 'Procedure ID saved  wrong')
        self.assertEqual(mat.timestamp, skm.procedure_date, 'Procedure ID saved  wrong')
