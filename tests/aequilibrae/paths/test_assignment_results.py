from unittest import TestCase
from aequilibrae.paths.results import AssignmentResults
import multiprocessing as mp


class TestAssignmentResults(TestCase):
    # def test_prepare(self):
    #     self.fail()
    #
    # def test_reset(self):
    #     self.fail()
    #
    # def test_total_flows(self):
    #     self.fail()

    def test_set_cores(self):
        a = AssignmentResults()
        a.set_cores(10)

        with self.assertRaises(ValueError):
            a.set_cores(1.3)

        a.set_cores(-2)
        self.assertEqual(a.cores, max(1, mp.cpu_count() - 2))

    def test_set_save_path_file(self):
        a = AssignmentResults()

        # Never save by default
        self.assertEqual(a.save_path_file, False)

    # def test_set_critical_links(self):
    #     self.fail()
    #
    # def test_get_load_results(self):
    #     self.fail()
    #
    # def test_save_to_disk(self):
    #     self.fail()
