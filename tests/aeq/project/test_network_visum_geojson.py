from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from aequilibrae import AequilibraeMatrix, TrafficAssignment, TrafficClass
from aequilibrae.project.network.visum_geojson_importer import (
    discover_visum_geojson_layers,
    inventory_visum_layers,
    parse_visum_capacity,
    parse_visum_length,
    parse_visum_speed,
    parse_visum_time,
    read_visum_geojson_layers,
)


def _write_layer(path: Path, records, crs="EPSG:4326"):
    gdf = gpd.GeoDataFrame(records, crs=crs)
    gdf.to_file(path, driver="GeoJSON")
    return gdf


@pytest.fixture
def visum_geojson_folder(tmp_path):
    folder = tmp_path / "visum"
    folder.mkdir()
    _write_layer(
        folder / "node.geojson",
        [
            {"NO": 1, "NAME": "A", "geometry": Point(0.0, 0.0)},
            {"NO": 2, "NAME": "B", "geometry": Point(0.01, 0.0)},
        ],
    )
    _write_layer(
        folder / "link.geojson",
        [
            {
                "NO": 100,
                "FROMNODENO": 1,
                "TONODENO": 2,
                "TSYSSET": "CAR,HGV",
                "R_TSYSSET": "CAR",
                "LC": "ARTERIAL",
                "R_LC": "LOCAL",
                "TYPENO": 10,
                "R_TYPENO": 20,
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
    _write_layer(
        folder / "zone_centroid.geojson",
        [
            {"NO": 1001, "NAME": "Z1", "geometry": Point(-0.01, 0.0)},
            {"NO": 1002, "NAME": "Z2", "geometry": Point(0.02, 0.0)},
        ],
    )
    _write_layer(
        folder / "zone_polygon.geojson",
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
    _write_layer(
        folder / "connector.geojson",
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
    _write_layer(
        folder / "countlocation.geojson",
        [
            {
                "NO": 5001,
                "LINKNO": 100,
                "FROMNODENO": 1,
                "TONODENO": 2,
                "CAR_ORIG": 950,
                "HVG_ORIG": 120,
                "MOTOR_ORIG": 1070,
                "DTVW": 1300,
                "CARS_LEFT": 10,
                "geometry": Point(0.005, 0.0),
            }
        ],
    )
    _write_layer(folder / "stop.geojson", [{"NO": 3001, "geometry": Point(0.0, 0.01)}])
    return folder


def test_discovery_inventory_and_deferred_layers(visum_geojson_folder):
    report = discover_visum_geojson_layers(visum_geojson_folder)

    assert not report.errors
    assert set(report.discovered_layers) == {
        "node",
        "link",
        "zone_centroid",
        "zone_polygon",
        "connector",
        "countlocation",
    }
    assert "stop" in report.deferred_layers

    layers, report = read_visum_geojson_layers(report.discovered_layers, report)
    inventory = inventory_visum_layers(layers)

    assert inventory["link"]["R_TSYSSET"]["role"] == "directional"
    assert inventory["countlocation"]["CARS_LEFT"]["role"] == "deferred"


def test_explicit_layer_mapping_and_missing_layer(visum_geojson_folder):
    report = discover_visum_geojson_layers(
        {
            "node": visum_geojson_folder / "node.geojson",
            "link": visum_geojson_folder / "link.geojson",
        }
    )

    assert {diag.layer for diag in report.errors} == {"connector", "zone_centroid"}


def test_unit_parsing():
    assert parse_visum_length("1.5km") == 1500
    assert parse_visum_length("25m") == 25
    assert parse_visum_speed("10m/s") == 36
    assert parse_visum_time("120s") == 2
    assert parse_visum_capacity("1800veh/h") == 1800

    with pytest.raises(ValueError):
        parse_visum_speed("10furlongs/h")


def test_crs_requires_explicit_assumption(monkeypatch, visum_geojson_folder):
    gdf = gpd.read_file(visum_geojson_folder / "node.geojson")
    gdf = gdf.set_crs(None, allow_override=True)
    monkeypatch.setattr(gpd, "read_file", lambda path: gdf)

    layers, report = read_visum_geojson_layers({"node": "node.geojson"})

    assert "node" in layers
    assert any(diag.code == "missing-crs" for diag in report.errors)

    _, report = read_visum_geojson_layers({"node": "node.geojson"}, accept_default_crs=True)

    assert not report.errors
    assert any(diag.code == "default-crs-assumed" for diag in report.diagnostics)


def test_create_from_visum_geojson_imports_network(empty_project, visum_geojson_folder):
    report = empty_project.network.create_from_visum_geojson(visum_geojson_folder)

    assert report.imported_counts == {"nodes": 2, "zones": 2, "links": 1, "connectors": 2}
    assert report.source_references["count_locations"] == [
        {
            "source_id": 5001,
            "link_id": 100,
            "counts": {"DTVW": 1300, "HVG_ORIG": 120, "MOTOR_ORIG": 1070, "CAR_ORIG": 950},
        }
    ]
    assert any(diag.code == "deferred-count-fields" for diag in report.diagnostics)

    with empty_project.db_connection as conn:
        assert conn.execute("select count(*) from nodes").fetchone()[0] == 4
        assert conn.execute("select count(*) from links").fetchone()[0] == 3
        assert conn.execute("select count(*) from zones").fetchone()[0] == 2
        assert conn.execute("select count(*) from modes where mode_id='h'").fetchone()[0] == 1
        assert conn.execute("select visum_length_ab from links where link_id=100").fetchone()[0] == 1000
        assert conn.execute("select a_node, b_node from links where link_id=100").fetchone() == (1, 2)
        assert conn.execute("select distance > 0 from links where link_id=100").fetchone()[0] == 1
        assert conn.execute("select modes is not null from nodes where node_id=1").fetchone()[0] == 1
        assert conn.execute("select link_types is not null from nodes where node_id=1").fetchone()[0] == 1
        assert conn.execute("select a_node, b_node from links where visum_connector_no=9001").fetchone() == (1001, 1)
        assert conn.execute("select modes from links where visum_connector_no=9001").fetchone()[0] == "ch"
        assert conn.execute("select count(*) from matrices").fetchone()[0] == 0

    empty_project.network.build_graphs(
        fields=["distance", "travel_time_ab", "travel_time_ba", "capacity_ab", "capacity_ba"], modes=["c"]
    )
    empty_project.network.set_time_field("travel_time")

    assert "c" in empty_project.network.graphs
    assert empty_project.network.graphs["c"].centroids.tolist() == [1001, 1002]
    assert empty_project.network.count_centroids() == 2

    matrix = AequilibraeMatrix()
    matrix.create_empty(zones=2, matrix_names=["demand"])
    matrix.index[:] = [1001, 1002]
    matrix.matrices[:, :, 0] = 0
    matrix.computational_view(["demand"])

    graph = empty_project.network.graphs["c"]
    graph.set_blocked_centroid_flows(False)
    traffic_class = TrafficClass("car", graph, matrix)
    assignment = TrafficAssignment(empty_project)
    assignment.add_class(traffic_class)
    assignment.set_time_field("travel_time")
    assignment.set_capacity_field("capacity")


def test_link_type_override(empty_project, visum_geojson_folder):
    empty_project.network.create_from_visum_geojson(visum_geojson_folder, link_type_mapping={"ARTERIAL": "arterial"})

    with empty_project.db_connection as conn:
        assert conn.execute("select link_type from links where link_id=100").fetchone()[0] == "arterial"


def test_mode_override_merges_hgv_into_car(empty_project, visum_geojson_folder):
    report = empty_project.network.create_from_visum_geojson(
        visum_geojson_folder, mode_mapping={"CAR": "c", "HGV": "c"}
    )

    with empty_project.db_connection as conn:
        modes = conn.execute("select modes from links where link_id=100").fetchone()[0]
        h_count = conn.execute("select count(*) from modes where mode_id='h'").fetchone()[0]

    assert modes == "c"
    assert h_count == 0
    assert report.mode_mapping == {"CAR": "c", "HGV": "c"}


def test_topology_validation_rejects_missing_node(tmp_path, visum_geojson_folder, empty_project):
    bad_link = gpd.read_file(visum_geojson_folder / "link.geojson")
    bad_link.loc[0, "TONODENO"] = 999
    bad_link.to_file(visum_geojson_folder / "link.geojson", driver="GeoJSON")

    with pytest.raises(ValueError, match="missing-node-reference"):
        empty_project.network.create_from_visum_geojson(visum_geojson_folder)


def test_topology_validation_rejects_endpoint_mismatch(visum_geojson_folder, empty_project):
    bad_link = gpd.read_file(visum_geojson_folder / "link.geojson")
    bad_link.loc[0, "geometry"] = LineString([(0.001, 0.0), (0.01, 0.0)])
    bad_link.to_file(visum_geojson_folder / "link.geojson", driver="GeoJSON")

    with pytest.raises(ValueError, match="endpoint-mismatch"):
        empty_project.network.create_from_visum_geojson(visum_geojson_folder)


def test_connector_validation_rejects_missing_zone(visum_geojson_folder, empty_project):
    bad_connector = gpd.read_file(visum_geojson_folder / "connector.geojson")
    bad_connector.loc[0, "ZONENO"] = 999
    bad_connector.to_file(visum_geojson_folder / "connector.geojson", driver="GeoJSON")

    with pytest.raises(ValueError, match="missing-zone-reference"):
        empty_project.network.create_from_visum_geojson(visum_geojson_folder)


def test_connector_validation_rejects_endpoint_mismatch(visum_geojson_folder, empty_project):
    bad_connector = gpd.read_file(visum_geojson_folder / "connector.geojson")
    bad_connector.loc[0, "geometry"] = LineString([(-0.011, 0.0), (0.0, 0.0)])
    bad_connector.to_file(visum_geojson_folder / "connector.geojson", driver="GeoJSON")

    with pytest.raises(ValueError, match="endpoint-mismatch"):
        empty_project.network.create_from_visum_geojson(visum_geojson_folder)


def test_invalid_units_are_reported_before_import(visum_geojson_folder, empty_project):
    bad_link = gpd.read_file(visum_geojson_folder / "link.geojson")
    bad_link.loc[0, "V0PRT"] = "10furlongs/h"
    bad_link.to_file(visum_geojson_folder / "link.geojson", driver="GeoJSON")

    with pytest.raises(ValueError, match="invalid-unit"):
        empty_project.network.create_from_visum_geojson(visum_geojson_folder)

    with empty_project.db_connection as conn:
        assert conn.execute("select count(*) from links").fetchone()[0] == 0


def test_non_positive_assignment_fields_are_diagnostic_warnings(visum_geojson_folder, empty_project):
    bad_link = gpd.read_file(visum_geojson_folder / "link.geojson")
    bad_link.loc[0, "CAPPRT"] = "0veh/h"
    bad_link.loc[0, "R_T0PRT"] = "-1min"
    bad_link.loc[0, "R_CAPPRT"] = None
    bad_link.to_file(visum_geojson_folder / "link.geojson", driver="GeoJSON")

    report = empty_project.network.create_from_visum_geojson(visum_geojson_folder)

    assert any(diag.code == "non-assignment-ready" and diag.field == "CAPPRT" for diag in report.diagnostics)
    assert any(diag.code == "non-assignment-ready" and diag.field == "R_T0PRT" for diag in report.diagnostics)
    assert any(diag.code == "non-assignment-ready" and diag.field == "R_CAPPRT" for diag in report.diagnostics)
