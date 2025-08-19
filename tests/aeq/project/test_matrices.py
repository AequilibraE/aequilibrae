import string
from math import floor
from os.path import join
from random import choice, randint
from shutil import copyfile

import pytest


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


def test_list(sioux_falls_example):
    matrices = sioux_falls_example.matrices
    df = matrices.list()
    with sioux_falls_example.db_connection as conn:
        cnt = conn.execute("select count(*) from Matrices").fetchone()[0]
    assert df.shape[0] == cnt, "Returned the wrong number of matrices in the database"
    assert df[df.status == "file missing"].shape[0] == 0, "Wrong # of records for missing matrix files"
