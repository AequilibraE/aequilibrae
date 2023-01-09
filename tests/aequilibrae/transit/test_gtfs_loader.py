from os.path import join, dirname, abspath
import pytest

import pandas as pd
from aequilibrae.project import Project
from aequilibrae.transit import Transit

from aequilibrae.transit.gtfs_loader import GTFSReader


class TestGTFSReader:
    @pytest.fixture
    def gtfs_loader(self, project: Project):
        Transit(project)
        return GTFSReader()

    @pytest.fixture
    def gtfs_fldr(self):
        return join(abspath(dirname("tests")), "tests/data/gtfs/gtfs_coquimbo.zip")

    def test_set_feed_path(self, gtfs_loader, gtfs_fldr):

        gtfs = gtfs_loader

        with pytest.raises(Exception):
            gtfs.set_feed_path(gtfs_fldr + "_")

    def test_load_data(self, gtfs_loader, gtfs_fldr):

        cap = pd.read_csv(join(abspath(dirname("tests")), "tests/data/gtfs/transit_max_speeds.txt"))

        df = cap[cap.city == "Coquimbo"]
        df.loc[df.min_distance < 100, "speed"] = 10
        dict_speeds = {x: df for x, df in df.groupby(["mode"])}
        gtfs = gtfs_loader

        gtfs._set_maximum_speeds(dict_speeds)
        gtfs.set_feed_path(gtfs_fldr)
        gtfs.load_data("2016-04-13")
