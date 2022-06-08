from unittest import TestCase

import numpy as np

from aequilibrae.distribution import GravityCalibration
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.parameters import Parameters
from ...data import siouxfalls_demand, siouxfalls_skims

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
    def setUp(self):
        # GravityCalibration requires an object that has a `parameters` attribute. `Parameters` fits
        # this requirement, so that we don't need to create a full project
        self.proj = Parameters()

    def test_calibrate(self):
        args = {"impedance": impedance, "matrix": matrix, "function": "power", "nan_to_zero": False}

        distributed_matrix = GravityCalibration(self.proj, **args)
        distributed_matrix.calibrate()
        if distributed_matrix.gap > 0.0001:
            self.fail("Calibration did not converge")

    def test_calibrate_with_omx(self):
        imped = AequilibraeMatrix()
        imped.load(siouxfalls_skims)
        imped.computational_view(["free_flow_time"])

        mat = AequilibraeMatrix()
        mat.load(siouxfalls_demand)
        mat.computational_view()

        args = {"impedance": imped, "matrix": mat, "function": "power", "nan_to_zero": False}

        distributed_matrix = GravityCalibration(self.proj, **args)
        distributed_matrix.calibrate()
        if distributed_matrix.gap > 0.0001:
            self.fail("Calibration did not converge")

        args = {"impedance": imped, "matrix": mat, "function": "power", "nan_to_zero": True}

        distributed_matrix = GravityCalibration(self.proj, **args)
        distributed_matrix.calibrate()
        if distributed_matrix.gap > 0.0001:
            self.fail("Calibration did not converge")
