from shapely.geometry import box

from aequilibrae.project.network.osm.osm_downloader import OSMDownloader


def test_do_work(mock_overpass_empty):
    _ = mock_overpass_empty
    o = OSMDownloader([box(0.0, 0.0, 0.1, 0.1)], ["car"])
    o.doWork()
    assert o.data["nodes"].empty and o.data["links"].empty, "It found links in the middle of the ocean"


def test_do_work2(mock_overpass_grid):
    _ = mock_overpass_grid
    # LITTLE PLACE IN THE MIDDLE OF THE Grand Canyon North Rim
    o = OSMDownloader([box(-112.185, 36.59, -112.179, 36.60)], ["car"])
    o.doWork()

    total_elements = len(o.data["nodes"]) + len(o.data["links"])
    assert total_elements <= 1000, "It found too many elements in the middle of the Grand Canyon"
    assert total_elements >= 10, "It found too few elements in the middle of the Grand Canyon"
