from unittest import TestCase
import tempfile
import os
from aequilibrae.project import Project
from aequilibrae.project.database_connection import database_connection
import uuid


class TestProject(TestCase):
    def setUp(self) -> None:
        self.temp_proj_folder = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex)
        self.proj = Project()
        self.proj.new(self.temp_proj_folder)

    def tearDown(self) -> None:
        self.proj.close()

    def test_opening_wrong_folder(self):
        temp_proj_folder = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex)
        self.proj.close()
        with self.assertRaises(FileNotFoundError):
            proj = Project()
            proj.open(temp_proj_folder)
        self.proj.open(self.temp_proj_folder)

    def test_creation(self):

        curr = self.proj.conn.cursor()
        curr.execute("""PRAGMA table_info(links);""")
        fields = curr.fetchall()
        fields = [x[1] for x in fields]

        if "distance" not in fields:
            self.fail("Table LINKS was not created correctly")

        curr = self.proj.conn.cursor()
        curr.execute("""PRAGMA table_info(nodes);""")
        nfields = curr.fetchall()
        nfields = [x[1] for x in nfields]

        if "is_centroid" not in nfields:
            self.fail("Table NODES was not created correctly")

    def test_close(self):

        _ = database_connection()

        self.proj.close()
        with self.assertRaises(FileNotFoundError):
            _ = database_connection()
