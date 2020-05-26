from unittest import TestCase
import os
from shutil import copytree, rmtree
import uuid
import random
import string
import tempfile
from sqlite3 import IntegrityError
from aequilibrae import Project
from aequilibrae.project.network.mode import Mode
from ...data import siouxfalls_project


class TestModes(TestCase):
    def setUp(self) -> None:
        self.temp_proj_folder = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex)
        copytree(siouxfalls_project, self.temp_proj_folder)
        self.proj = Project()
        self.proj.open(self.temp_proj_folder)
        self.curr = self.proj.conn.cursor()

    def tearDown(self) -> None:
        self.proj.close()
        rmtree(self.temp_proj_folder)

    def test_add(self):
        new_mode = Mode('F')
        name = [random.choice(string.ascii_letters + '_') for x in range(random.randint(1, 20))]
        name = ''.join(name)
        new_mode.mode_name = name
        self.proj.network.modes.add(new_mode)

        self.curr.execute('select mode_name from modes where mode_id="F"')
        self.assertEqual(self.curr.fetchone()[0], name, 'Could not save the mode properly to the database')

    def test_drop(self):
        self.proj.network.modes.drop('b')

        with self.assertRaises(IntegrityError):
            self.proj.network.modes.drop('c')

    def test_get(self):
        c = self.proj.network.modes.get('c')
        self.assertEqual('All motorized vehicles', c.description)
        del c

        with self.assertRaises(ValueError):
            _ = self.proj.network.modes.get('f')
