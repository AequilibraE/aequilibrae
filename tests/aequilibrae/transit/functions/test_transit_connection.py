import os
from tempfile import gettempdir
import unittest
from uuid import uuid4

from aequilibrae.project import Project
from aequilibrae.transit.functions.transit_connection import transit_connection


class TestTransitConnection(unittest.TestCase):
    def setUp(self) -> None:
        self.fldr = os.path.join(gettempdir(), uuid4().hex)
        self.prj = Project()
        self.prj.new(self.fldr)

    def tearDown(self) -> None:
        self.prj.close()

    def test_transit_connection_exception(self):
        fake_path = os.path.join(gettempdir(), uuid4().hex)
        with self.assertRaises(FileExistsError) as exception_context:
            transit_connection(fake_path)

        self.assertEqual(str(exception_context.exception), "")

    def test_transit_connection_none_path(self):
        cnx = transit_connection()

        self.assertGreater(
            len([x[0] for x in cnx.execute("SELECT name FROM sqlite_master WHERE type ='table'").fetchall()]), 1
        )

    def test_transit_connection(self):
        cnx = transit_connection(self.fldr)

        self.assertGreater(
            len([x[0] for x in cnx.execute("SELECT name FROM sqlite_master WHERE type ='table'").fetchall()]), 1
        )


if __name__ == "__name__":
    unittest.main()
