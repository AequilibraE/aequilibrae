from os.path import join, dirname, abspath
from tempfile import gettempdir
from unittest import TestCase
from uuid import uuid4

import pandas as pd
from aequilibrae.project import Project

from aequilibrae.transit.gtfs_loader import GTFSReader


class TestGTFSReader(TestCase):
    def setUp(self) -> None:
        self.fldr = join(gettempdir(), uuid4().hex)
        self.prj = Project()
        self.prj.new(self.fldr)

        self.gtfs_fldr = join(abspath(dirname("tests")), "tests/data/gtfs/2020-04-01.zip")

        self.cap = pd.read_csv(join(abspath(dirname("tests")), "tests/data/gtfs/transit_max_speeds.csv"))

        self.gtfs_loader = GTFSReader()

    def test_set_feed_path(self):
        # self.fail()
        self.gtfs_loader.set_feed_path(self.gtfs_fldr)

        with self.assertRaises(Exception):
            self.gtfs_loader.set_feed_path(self.gtfs_fldr + "_")

    def test_load_data(self):
        df = self.cap[self.cap.city == "Austin"]
        df.loc[df.min_distance < 100, "speed"] = 10
        dict_speeds = {x: df for x, df in df.groupby(["mode"])}
        self.gtfs_loader._set_maximum_speeds(dict_speeds)
        self.gtfs_loader.set_feed_path(self.gtfs_fldr)
        self.gtfs_loader.load_data("2020-04-01")
