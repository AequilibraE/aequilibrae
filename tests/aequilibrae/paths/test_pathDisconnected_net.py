from itertools import product

import geopandas as gpd

from aequilibrae.paths.results import PathResults


def test_path_disconnected_delete_link(sioux_falls_example):
    with sioux_falls_example.db_connection as conn:
        conn.executemany("delete from Links where link_id=?", [[2], [4], [5], [14]])

    sioux_falls_example.network.build_graphs()
    graph = sioux_falls_example.network.graphs["c"]
    graph.set_graph("free_flow_time")
    graph.set_blocked_centroid_flows(False)

    lonlat = graph.lonlat_index
    points = gpd.GeoSeries(gpd.points_from_xy(lonlat.lon, lonlat.lat), index=lonlat.index, crs=4326)
    points = points.to_crs(points.estimate_utm_crs())
    coordinates = points.get_coordinates()

    for early_exit, a_star in product([True, False], repeat=2):
        heuristic_scale = 1.0 if a_star else None
        result = PathResults(
            graph,
            1,
            5,
            early_exit=early_exit,
            a_star=a_star,
            heuristic_scale=heuristic_scale,
            coordinates=coordinates,
        )
        assert result.path is None, "Failed to return None for disconnected"
        result.compute_path(1, 2)
        assert len(result.path) == 1, "Returned the wrong thing for existing path on disconnected network"


def test_path_disconnected_penalize_link_in_memory(sioux_falls_example):
    links = [2, 4, 5, 14]

    sioux_falls_example.network.build_graphs()
    graph = sioux_falls_example.network.graphs["c"]
    graph.exclude_links(links)
    graph.set_graph("free_flow_time")
    graph.set_blocked_centroid_flows(False)

    lonlat = graph.lonlat_index
    points = gpd.GeoSeries(gpd.points_from_xy(lonlat.lon, lonlat.lat), index=lonlat.index, crs=4326)
    points = points.to_crs(points.estimate_utm_crs())
    coordinates = points.get_coordinates()

    for early_exit, a_star in product([True, False], repeat=2):
        heuristic_scale = 1.0 if a_star else None
        result = PathResults(
            graph,
            1,
            5,
            early_exit=early_exit,
            a_star=a_star,
            heuristic_scale=heuristic_scale,
            coordinates=coordinates,
        )
        assert result.path is None, "Failed to return None for disconnected"
        result.compute_path(1, 2)
        assert len(result.path) == 1, "Returned the wrong thing for existing path on disconnected network"
