import os
from shutil import copytree
import uuid
from tempfile import gettempdir
from unittest import TestCase
from ...data import siouxfalls_project
from aequilibrae.project import Project


class TestLog(TestCase):
    def setUp(self) -> None:
        os.environ['PATH'] = os.path.join(gettempdir(), 'temp_data') + ';' + os.environ['PATH']
        self.proj_dir = os.path.join(gettempdir(), uuid.uuid4().hex)
        copytree(siouxfalls_project, self.proj_dir)

        self.project = Project()
        self.project.open(self.proj_dir)

    def tearDown(self) -> None:
        self.project.close()

    def test_contents(self):
        log = self.project.log()
        cont = log.contents()
        self.assertEqual(len(cont), 4, 'Returned the wrong amount of data from the log')

    def test_clear(self):
        log = self.project.log()
        log.clear()

        with open(os.path.join(self.proj_dir, "aequilibrae.log"), 'r') as file:
            q = file.readlines()
        self.assertEqual(len(q), 0, 'Failed to clear the log file')
