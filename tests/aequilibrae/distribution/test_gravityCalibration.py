import os
import uuid
from shutil import copytree
from tempfile import gettempdir
from unittest import TestCase

import numpy as np

from aequilibrae.distribution import GravityCalibration
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.parameters import Parameters
from aequilibrae.project import Project
from ...data import siouxfalls_project

zones = 100

# Impedance matrix_procedures
name_test = AequilibraeMatrix().random_name()

args = {"file_name": name_test, "zones": zones, "matrix_names": ["impedance"]}

impedance = AequilibraeMatrix()
impedance.create_empty(**args)
impedance.matrices[:, :, 0] = np.random.rand(zones, zones) * 1000
impedance.index[:] = np.arange(impedance.zones) + 100
impedance.computational_view(["impedance"])

args["matrix_names"] = ["base_matrix"]
args["memory_only"] = True
matrix = AequilibraeMatrix()
matrix.create_empty(**args)
matrix.matrices[:, :, 0] = np.random.rand(zones, zones) * 1000
matrix.index[:] = np.arange(matrix.zones) + 100
matrix.computational_view(["base_matrix"])


class TestGravityCalibration(TestCase):
    def test_calibrate(self):
        par = Parameters()

        args = {"impedance": impedance, "matrix": matrix, "function": "power", "nan_to_zero": False}

        distributed_matrix = GravityCalibration(par, **args)
        distributed_matrix.calibrate()
        if distributed_matrix.gap > 0.0001:
            self.fail("Calibration did not converge")

    def test_calibrate_with_omx(self):
        self.proj_dir = os.path.join(gettempdir(), uuid.uuid4().hex)
        copytree(siouxfalls_project, self.proj_dir)
        self.proj = Project()
        self.proj.open(self.proj_dir)
        self.par = self.proj.project_parameters

        mats = self.proj.matrices
        mats.update_database()

        imped = mats.get_matrix("omx")
        imped.computational_view(["free_flow_time"])

        mat = mats.get_matrix("SiouxFalls_omx")
        mat.computational_view("matrix")

        args = {"impedance": imped, "matrix": mat, "function": "power", "nan_to_zero": False}

        distributed_matrix = GravityCalibration(self.par, **args)
        distributed_matrix.calibrate()
        if distributed_matrix.gap > 0.0001:
            self.fail("Calibration did not converge")

        args = {"impedance": imped, "matrix": mat, "function": "power", "nan_to_zero": True}

        distributed_matrix = GravityCalibration(self.par, **args)
        distributed_matrix.calibrate()
        if distributed_matrix.gap > 0.0001:
            self.fail("Calibration did not converge")
