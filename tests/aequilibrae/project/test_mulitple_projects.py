from unittest import TestCase
import tempfile
import os
from aequilibrae.context import get_active_project
from aequilibrae.project import Project
import uuid


class TestMultipleProjects(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proj = cls.new_project()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.proj.close()

    def setUp(self) -> None:
        self.proj.activate()

    @staticmethod
    def new_project():
        proj = Project()
        proj.new(os.path.join(tempfile.gettempdir(), uuid.uuid4().hex))
        return proj

    def test_current_project_is_active_project(self):
        self.assertEqual(self.proj, get_active_project())

    def test_switch_project(self):
        proj2 = self.new_project()
        self.assertIs(proj2, get_active_project())

    def test_reactivate_project(self):
        self.new_project()
        self.proj.activate()
        self.assertIs(self.proj, get_active_project())

    def test_raises_when_inactive(self):
        self.proj.deactivate()
        with self.assertRaises(FileNotFoundError):
            get_active_project()

    def test_close_project_deactivates(self):
        proj = self.new_project()
        proj.close()
        self.assertFalse(get_active_project(must_exist=False))

    def test_get_active_project_when_required(self):
        self.proj.deactivate()
        with self.assertRaises(FileNotFoundError):
            get_active_project()
