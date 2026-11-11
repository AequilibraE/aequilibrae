import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely import LineString, Point

from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit.lib_gtfs import GTFSRouteSystemBuilder
from aequilibrae.transit.functions.breaking_links_for_stop_access import split_links_at_stops
from aequilibrae.transit.route_map_matcher import RouteMapMatcher


@pytest.fixture
def map_matcher():
    points = [Point(-71.30, -29.90), Point(-71.29, -29.90), Point(-71.28, -29.90)]
    links = gpd.GeoDataFrame(
        {
            "link_id": [1, 2],
            "a_node": [10, 20],
            "b_node": [20, 30],
            "direction": [1, 1],
            "modes": ["c", "c"],
            "distance": [100.0, 200.0],
            "speed_ab": [10.0, 20.0],
            "speed_ba": [5.0, 10.0],
            "time_ab": [10.0, 10.0],
            "time_ba": [20.0, 20.0],
            "is_connector": [0, 0],
        },
        geometry=[LineString(points[:2]), LineString(points[1:])],
        crs=4326,
    )
    nodes = gpd.GeoDataFrame({"node_id": [10, 20, 30]}, geometry=points, crs=4326)
    stops = gpd.GeoDataFrame({"stop_id": ["first", "last"]}, geometry=[points[0], points[-1]], crs=4326)
    return RouteMapMatcher(links, nodes, stops)


@pytest.fixture
def broken_links(map_matcher):
    connectors = gpd.GeoDataFrame(
        {
            "link_id": [3, 4],
            "a_node": map_matcher.stops.stop_id.to_numpy(),
            "b_node": [10, 30],
            "direction": [0, 0],
            "modes": [None, None],
            "distance": [1.0, 1.0],
            "speed_ab": [1.0, 1.0],
            "speed_ba": [1.0, 1.0],
            "time_ab": [1.0, 1.0],
            "time_ba": [1.0, 1.0],
            "is_connector": [1, 1],
        },
        geometry=[LineString([point, point]) for point in map_matcher.stops.geometry],
        crs=map_matcher.links.crs,
    )
    return gpd.GeoDataFrame(pd.concat([map_matcher.links, connectors], ignore_index=True))


@pytest.mark.parametrize("direction", [-1, 1])
def test_reverse_graph_reverses_one_way_links(map_matcher, broken_links, direction):
    broken_links.loc[broken_links.is_connector == 0, "direction"] = direction
    map_matcher._RouteMapMatcher__graph_from_broken_net(broken_links)
    origin, destination = map(int, map_matcher.stops.stop_id)
    if direction == -1:
        origin, destination = destination, origin

    forward = map_matcher.graph.compute_path(origin, destination)
    reverse = map_matcher.reverse_graph.compute_path(destination, origin)
    assert forward.path is not None
    assert reverse.path is not None
    np.testing.assert_array_equal(reverse.path, forward.path[::-1])
    np.testing.assert_array_equal(reverse.path_nodes, forward.path_nodes[::-1])
    assert reverse.milepost[-1] == pytest.approx(forward.milepost[-1])
    assert map_matcher.graph.compute_path(destination, origin).path is None
    assert map_matcher.reverse_graph.compute_path(origin, destination).path is None


def test_reverse_graph_preserves_directional_costs(map_matcher, broken_links):
    broken_links["direction"] = 0
    map_matcher._RouteMapMatcher__graph_from_broken_net(broken_links)
    origin, destination = map(int, map_matcher.stops.stop_id)
    for graph, start, end, time in (
        (map_matcher.graph, origin, destination, 22.0),
        (map_matcher.reverse_graph, destination, origin, 22.0),
        (map_matcher.graph, destination, origin, 42.0),
        (map_matcher.reverse_graph, origin, destination, 42.0),
    ):
        result = graph.compute_path(start, end)
        assert result.path is not None
        end_index = np.flatnonzero(result.node_ids == end)[0]
        assert result.skims.matrices["time"][0, end_index] == pytest.approx(time)
        assert result.milepost[-1] == pytest.approx(302.0)

    attributes = ["direction", "speed_ab", "speed_ba", "time_ab", "time_ba"]
    pd.testing.assert_frame_equal(map_matcher._reverse_network.links[attributes], broken_links[attributes])


@pytest.mark.parametrize("mode", ["b", "t", None], ids=["bus", "rail", "no_modes"])
def test_graphs_retain_prefiltered_links_and_source_metadata(map_matcher, broken_links, mode):
    if mode is None:
        broken_links = broken_links.drop(columns="modes")
    else:
        broken_links.loc[broken_links.is_connector == 0, "modes"] = mode
    original = broken_links.copy(deep=True)
    map_matcher._RouteMapMatcher__graph_from_broken_net(broken_links)
    origin, destination = map(int, map_matcher.stops.stop_id)

    for graph, start, end in (
        (map_matcher.graph, origin, destination),
        (map_matcher.reverse_graph, destination, origin),
    ):
        assert set(graph.full_link_ids) == set(original.link_id)
        result = graph.compute_path(start, end)
        assert result.path is not None
        assert set(result.path) == set(original.link_id)

    pd.testing.assert_frame_equal(broken_links, original)
    pd.testing.assert_frame_equal(map_matcher._network_links, original)
    assert "modes" not in map_matcher._routing_network.links
    assert "modes" not in map_matcher._reverse_network.links


@pytest.fixture(scope="function")
def route_system_builder(build_gtfs_project):
    gtfs_file = build_gtfs_project.project.project_base_path / "gtfs_coquimbo.zip"
    with database_connection("transit") as transit_conn:
        yield GTFSRouteSystemBuilder(
            network=transit_conn,
            agency_identifier="LISERCO, LISANCO, LINCOSUR",
            file_path=gtfs_file,
            scenario=build_gtfs_project.scenario,
        )


def test_break_links_with_stops(route_system_builder):
    route_system_builder.load_date("2016-04-13")

    s = [[i, x.geo] for i, x in enumerate(route_system_builder.select_stops.values())]
    df = pd.DataFrame(s, columns=["stop_id", "geometry"])

    stops = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs(3857)
    links = route_system_builder.project.network.links.data.to_crs(3857)

    broken_links, new_nodes = split_links_at_stops(stops, links, tolerance=20)

    assert broken_links.geometry.length.sum() == pytest.approx(
        links.geometry.to_crs(broken_links.crs).length.sum(), abs=0.001
    )
    assert broken_links.shape[0] >= links.shape[0] + stops.shape[0]
    assert new_nodes.shape[0] >= stops.shape[0]
