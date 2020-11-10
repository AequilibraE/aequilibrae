from unittest import TestCase
import os
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
        self.curr = self.project.conn.cursor()

    def tearDown(self) -> None:
        self.project.close()

    def test_clear_database(self):
        self.curr.execute('Select count(*) from Matrices;')
        self.assertEqual(self.curr.fetchone()[0], 2, 'The test data started wrong')

        matrices = self.project.matrices

        matrices.clear_database()

        self.curr.execute('Select count(*) from Matrices;')
        self.assertEqual(self.curr.fetchone()[0], 1, 'Did not clear the database appropriately')

    def test_update_database(self):
        matrices = self.project.matrices

        matrices.update_database()


    # def test_list(self):
    #     self.fail()
    #
    # def test_get_matrix(self):
    #     self.fail()
