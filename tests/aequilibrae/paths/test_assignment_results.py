from unittest import TestCase
from aequilibrae.paths.results import AssignmentResults


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

        with self.assertRaises(ValueError):
            a.set_cores(-2)

    #
    # def test_set_critical_links(self):
    #     self.fail()
    #
    # def test_set_save_path_file(self):
    #     self.fail()
    #
    # def test_get_load_results(self):
    #     self.fail()
    #
    # def test_save_to_disk(self):
    #     self.fail()
