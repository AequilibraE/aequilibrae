import os
import shutil
from tempfile import gettempdir
import unittest
from uuid import uuid4

import pandas as pd
from aequilibrae.project import Project
from aequilibrae.project.database_connection import database_connection

from aequilibrae.transit.lib_gtfs import GTFSRouteSystemBuilder


class TestLibGTFS(unittest.TestCase):
    def setUp(self) -> None:
        self.fldr = os.path.join(gettempdir(), uuid4().hex)
        shutil.copytree(
            src=os.path.join(os.path.abspath(os.path.dirname("tests")), "tests/data/gtfs/AustinProject"),
            dst=self.fldr,
        )
        self.prj = Project()
        self.prj.open(self.fldr)

        self.prj.create_empty_transit()

        self.file_path = os.path.join(os.path.abspath(os.path.dirname("tests")), "tests/data/gtfs/2020-04-01.zip")
        self.network = database_connection(table_type="transit")

        self.system_builder = GTFSRouteSystemBuilder(
            network=self.network, agency_identifier="Capital Metro", file_path=self.file_path
        )

    def tearDown(self) -> None:
        self.prj.close()

    def test_set_capacities(self):
        self.system_builder.set_capacities({0: [150, 300, 300], 3: [42, 56, 56]})
        self.assertEqual(
            self.system_builder.gtfs_data.__dict__["__capacities__"], {0: [150, 300, 300], 3: [42, 56, 56]}
        )

    def test_dates_available(self):
        dates = self.system_builder.dates_available()
        self.assertEqual(type(dates), list)

    def test_set_allow_map_match(self):
        self.assertFalse(self.system_builder.__dict__["_GTFSRouteSystemBuilder__do_execute_map_matching"])
        self.system_builder.set_allow_map_match(True)
        self.assertTrue(self.system_builder.__dict__["_GTFSRouteSystemBuilder__do_execute_map_matching"])

    def test_map_match_tuple_exception(self):
        with self.assertRaises(TypeError) as exception_context:
            self.system_builder.map_match(route_types=3)

        self.assertEqual(str(exception_context.exception), "Route_types must be list or tuple")

    def test_map_match_int_exception(self):
        with self.assertRaises(TypeError) as exception_context:
            self.system_builder.map_match(route_types=[3.5])

        self.assertEqual(str(exception_context.exception), "All route types must be integers")

    def test_map_match(self):
        self.system_builder.load_date("2020-04-01")
        self.system_builder.set_allow_map_match(True)
        self.system_builder.map_match()
        self.system_builder.save_to_disk()

        self.assertGreater(self.network.execute("SELECT * FROM pattern_mapping;").fetchone()[0], 1)

    def test_set_agency_identifier(self):
        self.assertNotEqual(self.system_builder.gtfs_data.agency.agency, "CTA")
        self.system_builder.set_agency_identifier("CTA")
        self.assertEqual(self.system_builder.gtfs_data.agency.agency, "CTA")

    def test_set_feed(self):
        self.system_builder.set_feed(self.file_path)
        self.assertEqual(self.system_builder.gtfs_data.archive_dir, self.file_path)
        self.assertEqual(self.system_builder.gtfs_data.feed_date, "2020-04-01")

    def test_set_description(self):
        self.system_builder.set_description("CTA2019 fixed by John Doe after strong coffee")
        self.assertEqual(self.system_builder.description, "CTA2019 fixed by John Doe after strong coffee")

    def test_set_date(self):
        self.system_builder.set_date("2020-04-13")
        self.assertEqual(self.system_builder.__target_date__, "2020-04-13")

    def test_load_date(self):
        self.system_builder.load_date("2020-04-13")
        self.assertEqual(self.system_builder.gtfs_data.agency.service_date, "2020-04-13")
        self.assertTrue("1" in self.system_builder.select_routes.keys())

    def test_load_date_srid_exception(self):
        self.system_builder.srid = None
        with self.assertRaises(ValueError) as exception_context:
            self.system_builder.load_date("2020-04-01")

        self.assertEqual(str(exception_context.exception), "We cannot load data without an SRID")

    def test_load_date_not_available_date_exception(self):
        with self.assertRaises(ValueError) as exception_context:
            self.system_builder.load_date("2020-06-01")

        self.assertEqual(str(exception_context.exception), "The date chosen is not available in this GTFS feed")

    def test_set_do_raw_shapes(self):
        self.system_builder.set_do_raw_shapes(True)
        self.assertTrue(self.system_builder.__do_raw_shapes__)

    def test_create_raw_shapes(self):
        self.system_builder.load_date("2020-04-01")
        self.system_builder.create_raw_shapes()

        all_tables = [
            x[0] for x in self.network.execute("SELECT name FROM sqlite_master WHERE type ='table'").fetchall()
        ]
        self.assertTrue("raw_shapes" in all_tables)

    def test_save_to_disk(self):

        self.system_builder.load_date("2020-04-01")
        self.system_builder.save_to_disk()

        self.assertEqual(len(self.network.execute("SELECT * FROM route_links").fetchall()), 75)
        self.assertEqual(len(self.network.execute("SELECT * FROM trips;").fetchall()), 10)
        self.assertEqual(len(self.network.execute("SELECT * FROM routes;").fetchall()), 1)


if __name__ == "__name__":
    unittest.main()
