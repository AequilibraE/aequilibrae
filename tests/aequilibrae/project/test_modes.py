import os
import random
import string
import tempfile
import uuid
from shutil import copytree
from sqlite3 import IntegrityError
from unittest import TestCase

from aequilibrae import Project
from aequilibrae.project.network.mode import Mode
from aequilibrae.utils.db_utils import read_and_close
from ...data import siouxfalls_project


class TestModes(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(tempfile.gettempdir(), "temp_data") + ";" + os.environ["PATH"]
        self.temp_proj_folder = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex)
        copytree(siouxfalls_project, self.temp_proj_folder)
        self.proj = Project()
        self.proj.open(self.temp_proj_folder)

    def tearDown(self) -> None:
        self.proj.close()

    def test_add(self):
        new_mode = Mode("F", self.proj)
        name = [random.choice(string.ascii_letters + "_") for x in range(random.randint(1, 20))]
        name = "".join(name)
        new_mode.mode_name = name
        self.proj.network.modes.add(new_mode)

        with read_and_close(self.proj.path_to_file) as conn:
            mode = conn.execute('select mode_name from modes where mode_id="F"').fetchone()[0]

        self.assertEqual(mode, name, "Could not save the mode properly to the database")

    def test_drop(self):
        self.proj.network.modes.delete("b")

        with self.assertRaises(IntegrityError):
            self.proj.network.modes.delete("c")

    def test_get(self):
        c = self.proj.network.modes.get("c")
        self.assertEqual("All motorized vehicles", c.description)
        del c

        with self.assertRaises(ValueError):
            _ = self.proj.network.modes.get("f")

    def test_new(self):
        modes = self.proj.network.modes
        self.assertIsInstance(modes.new("h"), Mode, "Returned wrong type")

        m = list(modes.all_modes().keys())[0]
        with self.assertRaises(ValueError):
            modes.new(m)

    def test_fields(self):
        fields = self.proj.network.modes.fields
        fields.all_fields()
        self.assertEqual(fields._table, "modes", "Returned wrong table handler")
