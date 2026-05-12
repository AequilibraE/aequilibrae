import datetime
import os
import random
import uuid
from shutil import copyfile

import numpy as np
import openmatrix as omx
import pandas as pd
import pytest

from aequilibrae.matrix import AequilibraeMatrix

zones = 50


@pytest.fixture(scope="function")
def file_paths(sioux_falls_test):
    temp_folder = sioux_falls_test.project_base_path
    sf_skims = temp_folder / f"Aequilibrae_matrix_{uuid.uuid4()}.omx"
    copyfile(temp_folder / "matrices" / "sfalls_skims.omx", sf_skims)
    name_test = temp_folder / f"Aequilibrae_matrix_{uuid.uuid4()}.aem"
    copy_matrix_name = temp_folder / f"Aequilibrae_matrix_{uuid.uuid4()}.aem"
    csv_export_name = temp_folder / f"Aequilibrae_matrix_{uuid.uuid4()}.csv"
    omx_export_name = temp_folder / f"Aequilibrae_matrix_{uuid.uuid4()}.omx"

    return {
        "sf_skims": sf_skims,
        "name_test": name_test,
        "copy_matrix_name": copy_matrix_name,
        "csv_export_name": csv_export_name,
        "omx_export_name": omx_export_name,
    }


@pytest.fixture(scope="function")
def matrix(file_paths):
    args = {
        "file_name": file_paths["name_test"],
        "zones": zones,
        "matrix_names": ["mat", "seed", "dist"],
        "index_names": ["my_indices"],
        "memory_only": False,
    }

    matrix = AequilibraeMatrix()
    matrix.create_empty(**args)

    matrix.index[:] = np.arange(matrix.zones) + 100
    matrix.matrices[:, :, 0] = np.random.rand(matrix.zones, matrix.zones)
    matrix.matrices[:, :, 0] = matrix.mat * (1000 / np.sum(matrix.mat))
    matrix.setName("Test matrix - " + str(random.randint(1, 10)))
    matrix.setDescription("Generated at " + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))

    return matrix


def test_load(file_paths, matrix, no_index_omx):
    new_matrix = AequilibraeMatrix()
    # Cannot load OMX file with no indices
    with pytest.raises(LookupError):
        new_matrix.load(no_index_omx)

    new_matrix = AequilibraeMatrix()
    new_matrix.load(file_paths["name_test"])
    new_matrix.close()


def test_computational_view(matrix):
    matrix.computational_view(["mat", "seed"])
    matrix.mat.fill(0)
    matrix.seed.fill(0)
    assert matrix.matrix_view.shape[2] == 2, "Computational view returns the wrong number of matrices"

    matrix.computational_view(["mat"])
    matrix.matrix_view[:, :] = np.arange(zones**2).reshape(zones, zones)
    assert np.sum(matrix.mat) == np.sum(matrix.matrix_view), "Assigning to matrix view did not work"
    matrix.setName("Test matrix - " + str(random.randint(1, 10)))
    matrix.setDescription("Generated at " + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))


def test_computational_view_with_omx(omx_example):
    new_matrix = AequilibraeMatrix()
    new_matrix.load(omx_example)

    arrays = ["m1", "m2"]
    new_matrix.computational_view(arrays)
    total_mats = np.sum(new_matrix.matrix_view)

    new_matrix.computational_view([arrays[0]])
    total_m1 = np.sum(new_matrix.matrix_view)

    new_matrix.close()

    omx_file = omx.open_file(omx_example, "r")

    m1 = np.array(omx_file["m1"]).sum()
    m2 = np.array(omx_file["m2"]).sum()

    assert m1 + m2 == total_mats
    assert m1 == total_m1

    omx_file.close()


def test_copy(file_paths, matrix):
    # test in-memory matrix_procedures copy
    matrix_copy = matrix.copy(file_paths["copy_matrix_name"], cores=["mat"])

    assert np.array_equal(matrix_copy.mat, matrix.mat), "Matrix copy was not perfect"
    matrix_copy.close()


def test_copy_memory_only(file_paths, matrix):
    # test in-memory matrix_procedures copy
    matrix_copy = matrix.copy(file_paths["copy_matrix_name"], cores=["mat"], memory_only=True)

    for orig, new_arr in [[matrix_copy.mat, matrix.mat], [matrix_copy.index, matrix.index]]:
        assert np.array_equal(orig, new_arr), "Matrix copy was not perfect"
    matrix_copy.close()


def test_export_to_csv(file_paths, matrix):
    matrix.export(file_paths["csv_export_name"])
    df = pd.read_csv(file_paths["csv_export_name"])
    df.fillna(0, inplace=True)
    assert df.shape[0] == 2500, "Exported wrong size"
    assert df.shape[1] == 5, "Exported wrong size"
    assert abs(df.mat.sum() - np.nansum(matrix.matrices)) < 0.00001, "Exported wrong matrix total"


def test_export_to_omx(file_paths, matrix):
    matrix.export(file_paths["omx_export_name"])

    omxfile = omx.open_file(file_paths["omx_export_name"], "r")

    # Check if matrices values are compatible
    for m in matrix.names:
        sm = np.nansum(matrix.matrix[m])
        sm2 = np.nansum(np.array(omxfile[m]))

        assert sm == sm2, f"Matrix {m} was exported with the wrong value"
    omxfile.close()


def test_nan_to_num(matrix):
    m = matrix.mat.sum() - matrix.mat[1, 1]
    matrix.computational_view(["mat", "seed"])
    matrix.nan_to_num()
    matrix.matrices[1, 1, 0] = np.nan
    matrix.computational_view(["mat"])
    matrix.nan_to_num()

    assert abs(m - matrix.mat.sum()) <= 0.000000000001, "Total for mat matrix not maintained"


def test_copy_from_omx(omx_example):
    temp_file = AequilibraeMatrix().random_name()
    a = AequilibraeMatrix()
    a.create_from_omx(omx_path=omx_example, file_path=temp_file)

    omxfile = omx.open_file(omx_example, "r")

    # Check if matrices values are compatible
    for m in ["m1", "m2", "m3"]:
        sm = a.matrix[m].sum()
        sm2 = np.array(omxfile[m]).sum()
        assert sm == sm2, f"Matrix {m} was copied with the wrong value"

    assert np.all(a.index[:] == np.array(list(omxfile.mapping("taz").keys()))), "Index was not created properly"
    a.close()
    omxfile.close()

    if os.path.exists(temp_file):
        os.remove(temp_file)


def test_copy_from_omx_long_name(omx_example):
    temp_file = AequilibraeMatrix.random_name()
    a = AequilibraeMatrix()

    with pytest.raises(ValueError):
        a.create_from_omx(omx_path=omx_example, file_path=temp_file, robust=False)

    if os.path.exists(temp_file):
        os.remove(temp_file)


def test_copy_omx_wrong_content(omx_example):
    # Check if we get a result if we try to copy non-existing cores
    temp_file = AequilibraeMatrix().random_name()
    a = AequilibraeMatrix()

    with pytest.raises(ValueError):
        a.create_from_omx(temp_file, omx_example, cores=["m1", "m2", "m3", "m4"])

    with pytest.raises(ValueError):
        a.create_from_omx(temp_file, omx_example, mappings=["wrong index"])

    if os.path.exists(temp_file):
        os.remove(temp_file)


def test_get_matrix(file_paths):
    a = AequilibraeMatrix()
    a.load(file_paths["sf_skims"])

    with pytest.raises(AttributeError):
        a.get_matrix("does not exist")

    q = a.get_matrix("distance")
    assert q.shape[0] == 24


def test_save(file_paths, matrix):
    a = AequilibraeMatrix()
    a.load(file_paths["sf_skims"])

    a.computational_view(["distance"])
    new_mat = np.random.rand(a.zones, a.zones)
    a.matrix_view *= new_mat

    res = a.matrix_view.sum()

    a.save("new_name_for_matrix")
    assert res == a.matrix_view.sum(), "Saved wrong result"

    a.save(["new_name_for_matrix2"])
    assert a.view_names[0] == "new_name_for_matrix2", "Did not update computational view"
    assert len(a.view_names) == 1, "computational view with the wrong number of matrices"

    a.computational_view(["distance", "new_name_for_matrix"])

    with pytest.raises(ValueError):
        a.save(["just_one_name"])

    a.save(["one_name", "two_names"])

    with pytest.raises(ValueError):
        a.save("distance")

    b = AequilibraeMatrix()
    b.load(file_paths["name_test"])
    b.computational_view("seed")
    b.save()
    b.computational_view(["mat", "seed", "dist"])
    b.save()

    a.close()
    b.close()
