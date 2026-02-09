import pandas as pd
import geopandas as gpd
import pytest

from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit.lib_gtfs import GTFSRouteSystemBuilder
from aequilibrae.transit.functions.breaking_links_for_stop_access import split_links_at_stops


@pytest.fixture(scope="function")
def route_system_builder(build_gtfs_project):
    gtfs_file = build_gtfs_project.project.project_base_path / "gtfs_coquimbo.zip"
    with database_connection("transit") as transit_conn:
        yield GTFSRouteSystemBuilder(
            network=transit_conn, agency_identifier="LISERCO, LISANCO, LINCOSUR", file_path=gtfs_file
        )


def test_break_links_with_stops(route_system_builder):
    route_system_builder.load_date("2016-04-13")

    s = [[i, x.geo] for i, x in enumerate(route_system_builder.select_stops.values())]
    df = pd.DataFrame(s, columns=["stop_id", "geometry"])

    stops = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs(3857)
    links = route_system_builder.project.network.links.data.to_crs(3857)

    broken_links, new_nodes = split_links_at_stops(stops, links, tolerance=20)

    assert broken_links.geometry.length.sum() == pytest.approx(links.geometry.to_crs(broken_links.crs).length.sum(),
                                                               abs=0.001)
    assert broken_links.shape[0] >= links.shape[0] + stops.shape[0]
    assert  new_nodes.shape[0] >= stops.shape[0]
