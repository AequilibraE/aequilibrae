from unittest import TestCase
import tempfile
import os
from aequilibrae.project import Project
from aequilibrae.project.database_connection import database_connection
from aequilibrae import Parameters
import uuid
from functools import reduce


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

    def test_create_when_already_exists(self):
        with self.assertRaises(Exception):
            q = Project()
            q.new(os.path.join(tempfile.gettempdir(), uuid.uuid4().hex))

        with self.assertRaises(Exception):
            q = Project()
            q.open(os.path.join(tempfile.gettempdir(), uuid.uuid4().hex))

    def test_creation(self):
        p = Parameters().parameters["network"]

        curr = self.proj.conn.cursor()
        curr.execute("""PRAGMA table_info(links);""")
        fields = curr.fetchall()
        fields = [x[1] for x in fields]

        oneway = reduce(lambda a, b: dict(a, **b), p["links"]["fields"]["one-way"])
        owf = list(oneway.keys())
        twoway = reduce(lambda a, b: dict(a, **b), p["links"]["fields"]["two-way"])
        twf = []
        for k in list(twoway.keys()):
            twf.extend([f"{k}_ab", f"{k}_ba"])

        for f in owf + twf:
            if f not in fields:
                self.fail(f"Field {f} not added to links table")

        curr = self.proj.conn.cursor()
        curr.execute("""PRAGMA table_info(nodes);""")
        nfields = curr.fetchall()
        nfields = [x[1] for x in nfields]

        flds = reduce(lambda a, b: dict(a, **b), p["nodes"]["fields"])
        flds = list(flds.keys())

        for f in flds:
            if f not in nfields:
                self.fail(f"Field {f} not added to nodes table")

    def test_close(self):

        _ = database_connection()

        self.proj.close()
        with self.assertRaises(FileExistsError):
            _ = database_connection()
