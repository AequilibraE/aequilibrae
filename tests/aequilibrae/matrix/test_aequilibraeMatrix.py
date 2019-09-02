import datetime
import os
import random
import openmatrix as omx
from unittest import TestCase

import numpy as np
from tables.exceptions import NoSuchNodeError

from aequilibrae.matrix import AequilibraeMatrix
from ...data import omx_example

zones = 50
name_test = AequilibraeMatrix().random_name()
copy_matrix_name = AequilibraeMatrix().random_name()
csv_export_name = copy_matrix_name + ".csv"
omx_export_name = copy_matrix_name + ".omx"


class TestAequilibraeMatrix(TestCase):
    def test___init__(self):
        os.remove(name_test) if os.path.exists(name_test) else None
        args = {
            "file_name": name_test,
            "zones": zones,
            "matrix_names": ["mat", "seed", "dist"],
            "index_names": ["my indices"],
        }

        matrix = AequilibraeMatrix()
        matrix.create_empty(**args)

        matrix.index[:] = np.arange(matrix.zones) + 100
        matrix.mat[:, :] = np.random.rand(matrix.zones, matrix.zones)[:, :]
        matrix.mat[:, :] = matrix.mat[:, :] * (1000 / np.sum(matrix.mat[:, :]))
        matrix.setName("Test matrix - " + str(random.randint(1, 10)))
        matrix.setDescription("Generated at " + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
        matrix.close()
        del matrix

    def test_load(self):
        # self.test___init__()
        self.new_matrix = AequilibraeMatrix()
        self.new_matrix.load(name_test)

    def test_computational_view(self):
        self.test_load()
        self.new_matrix.computational_view(["mat", "seed"])
        self.new_matrix.mat.fill(0)
        self.new_matrix.seed.fill(0)
        if self.new_matrix.matrix_view.shape[2] != 2:
            self.fail("Computational view returns the wrong number of matrices")

        self.new_matrix.computational_view(["mat"])
        self.new_matrix.matrix_view[:, :] = np.arange(zones ** 2).reshape(zones, zones)
        if np.sum(self.new_matrix.mat) != np.sum(self.new_matrix.matrix_view):
            self.fail("Assigning to matrix view did not work")
        self.new_matrix.setName("Test matrix - " + str(random.randint(1, 10)))
        self.new_matrix.setDescription("Generated at " + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
        self.new_matrix.close()

    def test_copy(self):
        self.test_load()

        # test in-memory matrix_procedures copy

        matrix_copy = self.new_matrix.copy(copy_matrix_name, cores=["mat"])

        if not np.array_equal(matrix_copy.mat, self.new_matrix.mat):
            self.fail("Matrix copy was not perfect")
        matrix_copy.close()
        self.new_matrix.close()

    def test_export_to_csv(self):
        self.test_load()
        self.new_matrix.export(csv_export_name)
        self.new_matrix.close()

    def test_export_to_omx(self):
        self.test_load()
        self.new_matrix.export(omx_export_name)

        omxfile = omx.open_file(omx_export_name, "r")

        # Check if matrices values are compatible
        for m in self.new_matrix.names:
            sm = np.nansum(self.new_matrix.matrix[m])
            sm2 = np.nansum(np.array(omxfile[m]))
            if sm != sm2:
                self.fail("Matrix {} was exported with the wrong value".format(m))
        self.new_matrix.close()

    def test_nan_to_num(self):
        self.test_load()
        s = self.new_matrix.seed.sum() - self.new_matrix.seed[1, 1]
        m = self.new_matrix.mat.sum() - self.new_matrix.mat[1, 1]
        self.new_matrix.seed[1, 1] = np.nan
        self.new_matrix.computational_view(["mat", "seed"])
        self.new_matrix.nan_to_num()
        self.new_matrix.mat[1, 1] = np.nan
        self.new_matrix.computational_view(["mat"])
        self.new_matrix.nan_to_num()

        if s != self.new_matrix.seed.sum():
            self.fail("Total for seed matrix not maintained")

        if m != self.new_matrix.mat.sum():
            self.fail("Total for mat matrix not maintained")

    def test_copy_from_omx(self):
        temp_file = AequilibraeMatrix().random_name()
        a = AequilibraeMatrix()
        a.create_from_omx(temp_file, omx_example)

        omxfile = omx.open_file(omx_example, "r")

        # Check if matrices values are compatible
        for m in ["m1", "m2", "m3"]:
            sm = a.matrix[m].sum()
            sm2 = np.array(omxfile[m]).sum()
            if sm != sm2:
                self.fail("Matrix {} was copied with the wrong value".format(m))

        if np.any(a.index[:] != np.arange(a.zones)):
            self.fail("Index was not created properly")
        a.close()

    def test_copy_from_omx_long_name(self):

        temp_file = AequilibraeMatrix().random_name()
        a = AequilibraeMatrix()

        with self.assertRaises(ValueError):
            a.create_from_omx(temp_file, omx_example, robust=False)

    def test_copy_omx_wrong_content(self):
        # Check if we get a result if we try to copy non-existing cores
        temp_file = AequilibraeMatrix().random_name()
        a = AequilibraeMatrix()

        with self.assertRaises(ValueError):
            a.create_from_omx(temp_file, omx_example, cores=["m1", "m2", "m3", "m4"])

        with self.assertRaises(ValueError):
            a.create_from_omx(temp_file, omx_example, mappings=["wrong index"])
