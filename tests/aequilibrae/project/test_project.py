from unittest import TestCase
import tempfile
import os
from aequilibrae.project import Project
from aequilibrae import Parameters
import uuid
from functools import reduce

temp_proj_folder = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex)


class TestProject(TestCase):
    def test_creation(self):
        with self.assertRaises(FileNotFoundError):
            proj = Project()
            proj.load(temp_proj_folder)

        proj = Project()
        proj.new(temp_proj_folder)

        p = Parameters().parameters["network"]

        curr = proj.conn.cursor()
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

        curr = proj.conn.cursor()
        curr.execute("""PRAGMA table_info(nodes);""")
        nfields = curr.fetchall()
        nfields = [x[1] for x in nfields]

        flds = reduce(lambda a, b: dict(a, **b), p["nodes"]["fields"])
        flds = list(flds.keys())

        for f in flds:
            if f not in nfields:
                self.fail(f"Field {f} not added to nodes table")

        proj.close()
