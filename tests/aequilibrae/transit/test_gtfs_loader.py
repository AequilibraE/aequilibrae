import pytest

import pandas as pd

from aequilibrae.transit.gtfs_loader import GTFSReader


def test_set_feed_path(coquimbo_example):
    gtfs = GTFSReader()

    with pytest.raises(Exception):
        gtfs.set_feed_path(coquimbo_example.project_base_path / "wrong_name")


def test_load_data(build_gtfs_project, test_data_path):
    cap = pd.read_csv(test_data_path / "gtfs/transit_max_speeds.txt")

    df = cap[cap.city == "Coquimbo"]
    df.loc[df.min_distance < 100, "speed"] = 10
    dict_speeds = {x: df for x, df in df.groupby(["mode"])}  # noqa: C416
    gtfs = GTFSReader()

    gtfs._set_maximum_speeds(dict_speeds)
    gtfs.set_feed_path(build_gtfs_project.project_base_path / "gtfs_coquimbo.zip")
    gtfs.load_data("2016-04-13")
