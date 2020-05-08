from unittest import TestCase
import tempfile
import os
from aequilibrae.project import Project

temp_proj_name = os.path.join(tempfile.gettempdir(), "test_file.sqlite")


class TestProject(TestCase):
    def test_creation(self):
        if os.path.isfile(temp_proj_name):
            os.unlink(temp_proj_name)
        with self.assertRaises(FileNotFoundError):
            p = Project()
            p.load(temp_proj_name)

        p = Project()
        p.new(temp_proj_name)
        p.conn.close()
