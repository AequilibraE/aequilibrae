import string
from math import floor
from os.path import join
from pathlib import Path
from random import choice, randint
from shutil import copyfile

import numpy as np
import pytest

from aequilibrae.matrix import AequilibraeMatrix


def randomword(length):
    allowed_characters = string.ascii_letters + '_123456789@!()[]{};:"    -'
    val = "".join(choice(allowed_characters) for _ in range(length))
    if val[0] == "_" or val[-1] == "_":
        return randomword(length)
    return val


def mat_count(sioux_falls_example, should_have: int, error_message: str):
    with sioux_falls_example.db_connection as conn:
        cnt = conn.execute("Select count(*) from Matrices;").fetchone()[0]
    assert cnt == should_have, error_message


def test_set_record(sioux_falls_example):
    matrices = sioux_falls_example.matrices
    rec = matrices.get_record("demand_omx")
    with pytest.raises(ValueError):
        rec.name = "skims"
    with pytest.raises(ValueError):
        rec.file_name = "demand_mc.omx"
    assert rec.file_name == "demand.omx"
    assert rec.cores == 1, "Setting a file that exists did not correct the number of cores"


def test_clear_database(sioux_falls_test):
    matrices = sioux_falls_test.matrices
    mat_count(sioux_falls_test, 3, "The test data started wrong")
    matrices.clear_database()
    mat_count(sioux_falls_test, 2, "Did not clear the database appropriately")


def test_update_database(sioux_falls_test):
    matrices = sioux_falls_test.matrices
    mat_count(sioux_falls_test, 3, "The test data started wrong")
    matrices.update_database()
    mat_count(sioux_falls_test, 4, "Did not add to the database appropriately")
    rec = matrices.get_record("omx")
    existing = join(rec.fldr, rec.file_name)
    new_name = "test_name.omx"
    new_name1 = "test_name1.omx"
    copyfile(existing, join(rec.fldr, new_name))
    record = matrices.new_record("test_name1.omx", new_name)
    record.save()
    copyfile(existing, join(rec.fldr, new_name1))
    matrices.update_database()


def test_get_matrix(sioux_falls_example):
    matrices = sioux_falls_example.matrices
    with pytest.raises(Exception):
        _ = matrices.get_matrix("omxq")
    mat = matrices.get_matrix("demand_omx")
    mat.computational_view()
    assert floor(mat.matrix_view.sum()) == 360600, "Matrix loaded incorrectly"


def test_get_record(sioux_falls_example):
    matrices = sioux_falls_example.matrices
    rec = matrices.get_record("demand_mc")
    assert rec.cores == 3, "record populated wrong. Number of cores"
    assert rec.description is None, "record populated wrong. Description"


def test_record_update_cores(sioux_falls_test):
    matrices = sioux_falls_test.matrices
    rec = matrices.get_record("omx")
    rec.update_cores()
    assert rec.cores == 2, "Cores update did not work"


def test_save_record(sioux_falls_example):
    matrices = sioux_falls_example.matrices
    rec = matrices.get_record("demand_mc")
    text = randomword(randint(30, 100))
    rec.description = text
    rec.save()
    with sioux_falls_example.db_connection as conn:
        cnt = conn.execute('select description from matrices where name="demand_mc";').fetchone()[0]
    assert text == cnt, "Saving matrix record description failed"


def test_delete(sioux_falls_example):
    matrices = sioux_falls_example.matrices
    matrices.delete_record("demand_omx")
    with sioux_falls_example.db_connection as conn:
        cnt = conn.execute('select count(*) from matrices where name="demand_omx";').fetchone()[0]
    assert cnt == 0, "Deleting matrix record failed"
    with pytest.raises(Exception):
        matrices.get_record("demand_omx")


def test_import_file_omx(empty_project, omx_example):
    matrices = empty_project.matrices

    record = matrices.import_file(omx_example, name="imported_demand", file_name="imported_demand.omx")

    assert record.name == "imported_demand"
    assert record.file_name == "imported_demand.omx"
    assert record.cores == 4
    assert Path(matrices.fldr, "imported_demand.omx").is_file()

    mat = matrices.get_matrix("imported_demand")
    mat.computational_view(["m1"])
    assert floor(mat.matrix_view.sum()) == 46
    mat.close()


def test_import_file_aem(empty_project, tmp_path):
    source = tmp_path / "external_demand.aem"
    source_matrix = AequilibraeMatrix()
    source_matrix.create_empty(file_name=source, zones=2, matrix_names=["demand"], memory_only=False)
    source_matrix.index[:] = np.array([10, 20])
    source_matrix.matrix["demand"][:, :] = np.array([[1.0, 2.0], [3.0, 4.0]])
    source_matrix.close()

    matrices = empty_project.matrices
    record = matrices.import_file(source, name="external_demand")

    assert record.name == "external_demand"
    assert record.file_name == "external_demand.aem"
    assert record.cores == 1

    mat = matrices.get_matrix("external_demand")
    mat.computational_view(["demand"])
    assert mat.matrix_view.sum() == 10.0
    mat.close()


def test_import_file_rejects_duplicates(empty_project, omx_example):
    matrices = empty_project.matrices
    matrices.import_file(omx_example, name="imported_demand", file_name="imported_demand.omx")

    with pytest.raises(ValueError, match="already a matrix"):
        matrices.import_file(omx_example, name="imported_demand", file_name="other_demand.omx")

    with pytest.raises(ValueError, match="already a matrix record"):
        matrices.import_file(omx_example, name="other_demand", file_name="imported_demand.omx")


def test_import_file_rejects_unsupported_file(empty_project, tmp_path):
    matrices = empty_project.matrices
    unsupported = tmp_path / "demand.txt"
    unsupported.write_text("not a matrix")

    with pytest.raises(ValueError, match="Matrix needs to be either OMX or native AequilibraE"):
        matrices.import_file(unsupported)

    assert not Path(matrices.fldr, unsupported.name).exists()


def test_list(sioux_falls_example):
    matrices = sioux_falls_example.matrices
    df = matrices.list()
    with sioux_falls_example.db_connection as conn:
        cnt = conn.execute("select count(*) from Matrices").fetchone()[0]
    assert df.shape[0] == cnt, "Returned the wrong number of matrices in the database"
    assert df[df.status == "file missing"].shape[0] == 0, "Wrong # of records for missing matrix files"
