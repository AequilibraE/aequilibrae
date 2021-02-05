from math import floor
import string
from unittest import TestCase
from random import choice, randint
from shutil import copyfile
import os
from os.path import join
from shutil import copytree
import uuid
from tempfile import gettempdir
from ...data import siouxfalls_project
from aequilibrae.project import Project


class TestMatrices(TestCase):
    def setUp(self) -> None:
        proj_dir = os.path.join(gettempdir(), uuid.uuid4().hex)
        copytree(siouxfalls_project, proj_dir)

        self.project = Project()
        self.project.open(proj_dir)
        self.matrices = self.project.matrices
        self.matrices.reload()
        self.curr = self.project.conn.cursor()

    def tearDown(self) -> None:
        self.project.close()
        del self.project
        del self.matrices

    def randomword(self, length):
        allowed_characters = string.ascii_letters + '_123456789@!()[]{};:"    -'
        val = "".join(choice(allowed_characters) for i in range(length))
        if val[0] == "_" or val[-1] == "_":
            return self.randomword(length)
        return val

    def test_set_record(self):
        rec = self.matrices.get_record("omx2")
        with self.assertRaises(ValueError):
            rec.name = "omx"

        with self.assertRaises(ValueError):
            rec.file_name = "sfalls_skims.omx"

        rec.file_name = "SiouxFalls.omx"
        self.assertEqual(rec.cores, 1, "Setting a file that exists did not correct the number of cores")

    def test_clear_database(self):
        self.__mat_count(3, "The test data started wrong")

        self.matrices.clear_database()
        self.__mat_count(2, "Did not clear the database appropriately")

    def test_update_database(self):
        self.__mat_count(3, "The test data started wrong")

        self.matrices.update_database()
        self.__mat_count(4, "Did not add to the database appropriately")

        rec = self.matrices.get_record("omx")
        existing = join(rec.fldr, rec.file_name)
        new_name = "test_name.omx"
        new_name1 = "test_name1.omx"

        copyfile(existing, join(rec.fldr, new_name))
        record = self.matrices.new_record("test_name1.omx", new_name)
        record.save()

        copyfile(existing, join(rec.fldr, new_name1))
        self.matrices.update_database()

    def test_get_matrix(self):
        with self.assertRaises(Exception):
            _ = self.matrices.get_matrix("omxq")

        mat = self.matrices.get_matrix("oMx")
        mat.computational_view()
        self.assertEqual(floor(mat.matrix_view.sum()), 20309, "Matrix loaded incorrectly")

    def test_get_record(self):
        rec = self.matrices.get_record("omx")
        self.assertEqual(rec.cores, 4, "record populated wrong. Number of cores")
        self.assertEqual(rec.description, None, "record populated wrong. Description")

    def test_record_update_cores(self):
        rec = self.matrices.get_record("omx")
        rec.update_cores()
        self.assertEqual(rec.cores, 2, "Cores update did not work")

    def test_save_record(self):
        rec = self.matrices.get_record("omx")

        text = self.randomword(randint(30, 100))
        rec.description = text
        rec.save()

        self.curr.execute('select description from matrices where name="omx";')
        self.assertEqual(text, self.curr.fetchone()[0], "Saving matrix record description failed")

    def test_delete(self):
        self.matrices.delete_record("omx")
        self.curr.execute('select count(*) from matrices where name="omx";')

        self.assertEqual(0, self.curr.fetchone()[0], " Deleting matrix record failed")

        with self.assertRaises(Exception):
            self.matrices.get_record("omx")

    def test_list(self):
        df = self.matrices.list()
        self.curr.execute("select count(*) from Matrices")
        self.assertEqual(df.shape[0], self.curr.fetchone()[0], "Returned the wrong number of matrices in the database")

        self.assertEqual(df[df.status == "file missing"].shape[0], 1, "Wrong # of records for missing matrix files")

    def __mat_count(self, should_have: int, error_message: str) -> None:
        self.curr.execute("Select count(*) from Matrices;")
        self.assertEqual(self.curr.fetchone()[0], should_have, error_message)
