from os.path import join, dirname, abspath
from pathlib import Path

import pytest

import pandas as pd
from aequilibrae.transit import Transit

from aequilibrae.transit.gtfs_loader import GTFSReader
from aequilibrae.utils.create_example import create_example


@pytest.fixture
def gtfs_loader(create_gtfs_project):
    return GTFSReader()


@pytest.fixture
def gtfs_fldr(create_path):
    return join(create_path, "gtfs_coquimbo.zip")


def test_set_feed_path(gtfs_loader, gtfs_fldr):
    gtfs = gtfs_loader

    with pytest.raises(Exception):
        gtfs.set_feed_path(gtfs_fldr + "_")


def test_load_data(gtfs_loader, gtfs_fldr):
    pth = Path(__file__).parent.parent.parent
    cap = pd.read_csv(pth / "data/gtfs/transit_max_speeds.txt")

    df = cap[cap.city == "Coquimbo"]
    df.loc[df.min_distance < 100, "speed"] = 10
    dict_speeds = {x: df for x, df in df.groupby(["mode"])}
    gtfs = gtfs_loader

    gtfs._set_maximum_speeds(dict_speeds)
    gtfs.set_feed_path(gtfs_fldr)
    gtfs.load_data("2016-04-13")
