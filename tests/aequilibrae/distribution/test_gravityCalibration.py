from unittest import TestCase

import numpy as np

from aequilibrae.distribution import GravityCalibration
from aequilibrae.matrix import AequilibraeMatrix

zones = 100

# Impedance matrix_procedures
name_test = AequilibraeMatrix().random_name()

args = {"file_name": name_test, "zones": zones, "matrix_names": ["impedance"]}

impedance = AequilibraeMatrix()
impedance.create_empty(**args)
impedance.impedance[:, :] = np.random.rand(zones, zones)[:, :] * 1000
impedance.index[:] = np.arange(impedance.zones) + 100
impedance.computational_view(["impedance"])

args["matrix_names"] = ["base_matrix"]
args["file_name"] = AequilibraeMatrix().random_name()
matrix = AequilibraeMatrix()
matrix.create_empty(**args)
matrix.base_matrix[:, :] = np.random.rand(zones, zones)[:, :] * 1000
matrix.index[:] = np.arange(matrix.zones) + 100
matrix.computational_view(["base_matrix"])


class TestGravityCalibration(TestCase):
    def test_calibrate(self):
        args = {"impedance": impedance, "matrix": matrix, "function": "power", "nan_to_zero": False}

        distributed_matrix = GravityCalibration(**args)
        distributed_matrix.calibrate()
        if distributed_matrix.gap > 0.0001:
            self.fail("Calibration did not converge")

    def test_check_inputs(self):
        # self.fail()
        pass

    def test_apply_gravity(self):
        # self.fail()
        pass

    def test_get_parameters(self):
        # self.fail()
        pass
