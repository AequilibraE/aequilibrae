import datetime
import os
import random
from os.path import join
from shutil import copyfile
import openmatrix as omx
from unittest import TestCase
from tempfile import gettempdir
import numpy as np
import uuid
from aequilibrae.matrix import AequilibraeMatrix
from ...data import omx_example, no_index_omx, siouxfalls_skims
import pandas as pd
import pytest

zones = 50


class TestAequilibraeMatrix(TestCase):
    matrix = None

    def setUp(self) -> None:
        temp_folder = gettempdir()
        self.sf_skims = join(temp_folder, f"Aequilibrae_matrix_{uuid.uuid4()}.omx")
        copyfile(siouxfalls_skims, self.sf_skims)
        self.name_test = join(temp_folder, f"Aequilibrae_matrix_{uuid.uuid4()}.aem")
        self.copy_matrix_name = join(temp_folder, f"Aequilibrae_matrix_{uuid.uuid4()}.aem")
        self.csv_export_name = join(temp_folder, f"Aequilibrae_matrix_{uuid.uuid4()}.csv")
        self.omx_export_name = join(temp_folder, f"Aequilibrae_matrix_{uuid.uuid4()}.omx")

        if self.matrix is not None:
            return
        args = {
            "file_name": self.name_test,
            "zones": zones,
            "matrix_names": ["mat", "seed", "dist"],
            "index_names": ["my_indices"],
            "memory_only": False,
        }

        self.matrix = AequilibraeMatrix()
        self.matrix.create_empty(**args)

        self.matrix.index[:] = np.arange(self.matrix.zones) + 100
        self.matrix.matrices[:, :, 0] = np.random.rand(self.matrix.zones, self.matrix.zones)
        self.matrix.matrices[:, :, 0] = self.matrix.mat * (1000 / np.sum(self.matrix.mat))
        self.matrix.setName("Test matrix - " + str(random.randint(1, 10)))
        self.matrix.setDescription("Generated at " + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
        self.new_matrix = self.matrix

    def tearDown(self) -> None:
        try:
            del self.matrix
            os.remove(self.name_test) if os.path.exists(self.name_test) else None
            os.remove(self.csv_export_name) if os.path.exists(self.csv_export_name) else None
            os.remove(self.copy_matrix_name) if os.path.exists(self.copy_matrix_name) else None
            os.remove(self.omx_export_name) if os.path.exists(self.omx_export_name) else None
        except Exception as e:
            print(f"Could not delete.  {e.args}")

    def test_load(self):
        self.new_matrix = AequilibraeMatrix()
        # Cannot load OMX file with no indices
        with self.assertRaises(LookupError):
            self.new_matrix.load(no_index_omx)

        self.new_matrix = AequilibraeMatrix()
        self.new_matrix.load(self.name_test)
        del self.new_matrix

    def test_computational_view(self):
        self.new_matrix.computational_view(["mat", "seed"])
        self.new_matrix.mat.fill(0)
        self.new_matrix.seed.fill(0)
        if self.new_matrix.matrix_view.shape[2] != 2:
            self.fail("Computational view returns the wrong number of matrices")

        self.new_matrix.computational_view(["mat"])
        self.new_matrix.matrix_view[:, :] = np.arange(zones**2).reshape(zones, zones)
        if np.sum(self.new_matrix.mat) != np.sum(self.new_matrix.matrix_view):
            self.fail("Assigning to matrix view did not work")
        self.new_matrix.setName("Test matrix - " + str(random.randint(1, 10)))
        self.new_matrix.setDescription("Generated at " + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
        del self.new_matrix

    def test_computational_view_with_omx(self):
        self.new_matrix = AequilibraeMatrix()
        self.new_matrix.load(omx_example)

        arrays = ["m1", "m2"]
        self.new_matrix.computational_view(arrays)
        total_mats = np.sum(self.new_matrix.matrix_view)

        self.new_matrix.computational_view([arrays[0]])
        total_m1 = np.sum(self.new_matrix.matrix_view)

        self.new_matrix.close()

        omx_file = omx.open_file(omx_example, "r")

        m1 = np.array(omx_file["m1"]).sum()
        m2 = np.array(omx_file["m2"]).sum()

        self.assertEqual(m1 + m2, total_mats)
        self.assertEqual(m1, total_m1)

        omx_file.close()
        del omx_file

    def test_copy(self):
        # test in-memory matrix_procedures copy

        matrix_copy = self.new_matrix.copy(self.copy_matrix_name, cores=["mat"])

        if not np.array_equal(matrix_copy.mat, self.new_matrix.mat):
            self.fail("Matrix copy was not perfect")
        matrix_copy.close()
        del matrix_copy

    def test_copy_memory_only(self):
        # test in-memory matrix_procedures copy

        matrix_copy = self.new_matrix.copy(self.copy_matrix_name, cores=["mat"], memory_only=True)

        for orig, new_arr in [[matrix_copy.mat, self.new_matrix.mat], [matrix_copy.index, self.new_matrix.index]]:
            if not np.array_equal(orig, new_arr):
                self.fail("Matrix copy was not perfect")
        matrix_copy.close()

    def test_export_to_csv(self):
        self.new_matrix.export(self.csv_export_name)
        df = pd.read_csv(self.csv_export_name)
        df.fillna(0, inplace=True)
        self.assertEqual(df.shape[0], 2500, "Exported wrong size")
        self.assertEqual(df.shape[1], 5, "Exported wrong size")
        self.assertAlmostEqual(df.mat.sum(), np.nansum(self.new_matrix.matrices), 5, "Exported wrong matrix total")

    def test_export_to_omx(self):
        self.new_matrix.export(self.omx_export_name)

        omxfile = omx.open_file(self.omx_export_name, "r")

        # Check if matrices values are compatible
        for m in self.new_matrix.names:
            sm = np.nansum(self.new_matrix.matrix[m])
            sm2 = np.nansum(np.array(omxfile[m]))

            self.assertEqual(sm, sm2, "Matrix {} was exported with the wrong value".format(m))
        del omxfile

    def test_nan_to_num(self):
        m = self.new_matrix.mat.sum() - self.new_matrix.mat[1, 1]
        self.new_matrix.computational_view(["mat", "seed"])
        self.new_matrix.nan_to_num()
        self.new_matrix.matrices[1, 1, 0] = np.nan
        self.new_matrix.computational_view(["mat"])
        self.new_matrix.nan_to_num()

        if abs(m - self.new_matrix.mat.sum()) > 0.000000000001:
            self.fail("Total for mat matrix not maintained")
        del self.new_matrix

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

        if np.any(a.index[:] != np.array(list(omxfile.mapping("taz").keys()))):
            self.fail("Index was not created properly")
        a.close()
        del a
        del omxfile

    def test_copy_from_omx_long_name(self):
        temp_file = AequilibraeMatrix().random_name()
        a = AequilibraeMatrix()

        with self.assertRaises(ValueError):
            a.create_from_omx(temp_file, omx_example, robust=False)
        del a

    def test_copy_omx_wrong_content(self):
        # Check if we get a result if we try to copy non-existing cores
        temp_file = AequilibraeMatrix().random_name()
        a = AequilibraeMatrix()

        with self.assertRaises(ValueError):
            a.create_from_omx(temp_file, omx_example, cores=["m1", "m2", "m3", "m4"])

        with self.assertRaises(ValueError):
            a.create_from_omx(temp_file, omx_example, mappings=["wrong index"])
        del a

    def test_get_matrix(self):
        a = AequilibraeMatrix()
        a.load(self.sf_skims)

        with self.assertRaises(AttributeError):
            a.get_matrix("does not exist")

        q = a.get_matrix("distance")
        self.assertEqual(q.shape[0], 24)

        a = AequilibraeMatrix()
        a.load(self.name_test)
        print(np.array_equal(a.get_matrix("seed"), a.matrix["seed"]))

        del a

    def test_save(self):
        a = AequilibraeMatrix()
        a.load(self.sf_skims)

        a.computational_view(["distance"])
        new_mat = np.random.rand(a.zones, a.zones)
        a.matrix_view *= new_mat

        res = a.matrix_view.sum()

        a.save("new_name_for_matrix")
        self.assertEqual(res, a.matrix_view.sum(), "Saved wrong result")

        a.save(["new_name_for_matrix2"])
        self.assertEqual(a.view_names[0], "new_name_for_matrix2", "Did not update computational view")
        self.assertEqual(len(a.view_names), 1, "computational view with the wrong number of matrices")

        a.computational_view(["distance", "new_name_for_matrix"])

        with self.assertRaises(ValueError):
            a.save(["just_one_name"])

        a.save(["one_name", "two_names"])

        with self.assertRaises(ValueError):
            a.save("distance")

        b = AequilibraeMatrix()
        b.load(self.name_test)
        b.computational_view("seed")
        b.save()
        b.computational_view(["mat", "seed", "dist"])
        b.save()


def test_can_reuse_matrix_after_close(tmp_path):
    kwargs = {
        "file_name": tmp_path / "matrix.aem",
        "zones": 1,
        "matrix_names": ["mat"],
        "index_names": ["index"],
    }
    matrix = AequilibraeMatrix()
    matrix.create_empty(**kwargs)
    matrix.close()
    try:
        AequilibraeMatrix().create_empty(**kwargs)
    except OSError:
        pytest.fail("Should not raise OSError")


def test_matrix_reference_doesnt_prevent_resource_cleanup(tmp_path):
    kwargs = {
        "file_name": tmp_path / "matrix.aem",
        "zones": 1,
        "matrix_names": ["mat"],
        "index_names": ["index"],
    }
    matrix = AequilibraeMatrix()
    matrix.create_empty(**kwargs)
    _ = matrix.mat
    del matrix

    try:
        AequilibraeMatrix().create_empty(**kwargs)
    except OSError:
        pytest.fail("Should not raise OSError")
