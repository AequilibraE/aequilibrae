import os
from tempfile import gettempdir
import unittest
from uuid import uuid4
from aequilibrae.project import Project
from aequilibrae.transit import Transit
from aequilibrae.utils.create_example import create_example


class TestTransit(unittest.TestCase):
    def setUp(self) -> None:
        self.fldr = os.path.join(gettempdir(), uuid4().hex)
        self.prj = Project()
        self.prj.new(self.fldr)

    def test_new_gtfs(self):
        data = Transit(self.prj)
        transit = data.new_gtfs(
            agency="",
            file_path=os.path.join(os.path.abspath(os.path.dirname("tests")), "tests/data/gtfs/2020-04-01.zip"),
        )

        self.assertEqual(str(type(transit)), "<class 'aequilibrae.transit.lib_gtfs.GTFSRouteSystemBuilder'>")

    def test__check_connection(self):
        example = create_example(os.path.join(gettempdir(), uuid4().hex), "nauru")

        with self.assertRaises(FileNotFoundError) as exception_context:
            Transit(example)

        self.assertEqual(
            str(exception_context.exception),
            "Public Transport model does not exist. Create a new one or change your path.",
        )

    def test_create_empty_transit_exception(self):
        with self.assertRaises(FileExistsError) as exception_context:
            self.prj.create_empty_transit()

        self.assertEqual(str(exception_context.exception), "Public Transport database already exists.")

    def test_create_empty_transit(self):
        temp_path = os.path.join(gettempdir(), uuid4().hex)
        example = create_example(temp_path, "nauru")
        example.create_empty_transit()

        self.assertTrue(os.path.isfile(os.path.join(temp_path, "public_transport.sqlite")))


if __name__ == "__name__":
    unittest.main()
