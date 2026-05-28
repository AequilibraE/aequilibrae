"""
.. _import_from_visum_geojson:

Importing a network from VISUM GeoJSON
======================================

This example creates a tiny VISUM-like GeoJSON export locally and imports it as an
AequilibraE private-traffic network.
"""

from tempfile import TemporaryDirectory
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon

from aequilibrae import Project


def write_layer(path, records):
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf.to_file(path, driver="GeoJSON")


with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
    project = Project()
    project.new(Path(temp_dir) / "visum_project")

    visum_dir = project.project_base_path / "visum_geojson"
    visum_dir.mkdir()

    write_layer(
        visum_dir / "node.geojson",
        [
            {"NO": 1, "NAME": "A", "geometry": Point(0.0, 0.0)},
            {"NO": 2, "NAME": "B", "geometry": Point(0.01, 0.0)},
        ],
    )
    write_layer(
        visum_dir / "link.geojson",
        [
            {
                "NO": 100,
                "FROMNODENO": 1,
                "TONODENO": 2,
                "TSYSSET": "CAR,HGV",
                "R_TSYSSET": "CAR",
                "LC": "ARTERIAL",
                "LENGTH": "1km",
                "R_LENGTH": "1.1km",
                "V0PRT": "60km/h",
                "R_V0PRT": "55km/h",
                "CAPPRT": "1200veh/h",
                "R_CAPPRT": "1100veh/h",
                "geometry": LineString([(0.0, 0.0), (0.01, 0.0)]),
            }
        ],
    )
    write_layer(
        visum_dir / "zone_centroid.geojson",
        [
            {"NO": 1001, "NAME": "Z1", "geometry": Point(-0.01, 0.0)},
            {"NO": 1002, "NAME": "Z2", "geometry": Point(0.02, 0.0)},
        ],
    )
    write_layer(
        visum_dir / "zone_polygon.geojson",
        [
            {
                "NO": 1001,
                "NAME": "Z1",
                "geometry": Polygon([(-0.02, -0.01), (-0.005, -0.01), (-0.005, 0.01), (-0.02, 0.01)]),
            },
            {
                "NO": 1002,
                "NAME": "Z2",
                "geometry": Polygon([(0.015, -0.01), (0.03, -0.01), (0.03, 0.01), (0.015, 0.01)]),
            },
        ],
    )
    write_layer(
        visum_dir / "connector.geojson",
        [
            {
                "NO": 9001,
                "ZONENO": 1001,
                "NODENO": 1,
                "TSYSSET": "CAR,HGV",
                "R_TSYSSET": "CAR,HGV",
                "LENGTH": "100m",
                "R_LENGTH": "100m",
                "V0PRT": "30km/h",
                "R_V0PRT": "30km/h",
                "CAPPRT": "9999veh/h",
                "R_CAPPRT": "9999veh/h",
                "geometry": LineString([(-0.01, 0.0), (0.0, 0.0)]),
            },
            {
                "NO": 9002,
                "ZONENO": 1002,
                "NODENO": 2,
                "TSYSSET": "CAR,HGV",
                "R_TSYSSET": "CAR,HGV",
                "LENGTH": "100m",
                "R_LENGTH": "100m",
                "V0PRT": "30km/h",
                "R_V0PRT": "30km/h",
                "CAPPRT": "9999veh/h",
                "R_CAPPRT": "9999veh/h",
                "geometry": LineString([(0.02, 0.0), (0.01, 0.0)]),
            },
        ],
    )
    write_layer(
        visum_dir / "countlocation.geojson",
        [
            {
                "NO": 5001,
                "LINKNO": 100,
                "FROMNODENO": 1,
                "TONODENO": 2,
                "CAR_ORIG": 950,
                "HVG_ORIG": 120,
                "geometry": Point(0.005, 0.0),
            }
        ],
    )

    report = project.network.create_from_visum_geojson(visum_dir)
    project.network.build_graphs(
        fields=["distance", "travel_time_ab", "travel_time_ba", "capacity_ab", "capacity_ba"], modes=["c"]
    )
    project.network.set_time_field("travel_time")

    print(report.imported_counts)

    project.close()
