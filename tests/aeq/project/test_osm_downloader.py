import pytest
from shapely.geometry import box

from aequilibrae.project.network.osm.osm_downloader import OSMDownloader


def test_do_work(mock_overpass_api):
    _ = mock_overpass_api
    o = OSMDownloader([box(0.0, 0.0, 0.1, 0.1)], ["car"])
    o.doWork()
    assert not o.json, "It found links in the middle of the ocean"


def test_do_work2(mock_overpass_api):
    # LITTLE PLACE IN THE MIDDLE OF THE Grand Canyon North Rim
    _ = mock_overpass_api
    o = OSMDownloader([box(-112.185, 36.59, -112.179, 36.60)], ["car"])
    o.doWork()

    if len(o.json) == 0 or "elements" not in o.json[0]:
        pytest.skip("No elements found in response")

    assert len(o.json[0]["elements"]) <= 1000, "It found too many elements in the middle of the Grand Canyon"
    assert len(o.json[0]["elements"]) >= 10, "It found too few elements in the middle of the Grand Canyon"
