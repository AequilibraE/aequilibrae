import uuid
import os
import tempfile
from unittest import TestCase
from aequilibrae.context import activate_project
from aequilibrae.project.database_connection import database_connection
from aequilibrae.project import Project


class TestDatabaseConnection(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(tempfile.gettempdir(), "temp_data") + ";" + os.environ["PATH"]
        activate_project(None)

    def test_database_connection(self):
        # Errors when project does not exist
        with self.assertRaises(FileNotFoundError):
            _ = database_connection()

    def test_connection_with_new_project(self):
        temp_proj_folder = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex)
        proj = Project()
        proj.new(temp_proj_folder)
        proj.close()

        proj = Project()
        proj.open(temp_proj_folder)
        conn = database_connection()
        cursor = conn.cursor()
        cursor.execute("select count(*) from links")

        self.assertEqual(cursor.fetchone()[0], 0, "Returned more links thant it should have")
        proj.close()
