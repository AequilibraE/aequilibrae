from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from aequilibrae import AequilibraeMatrix, TrafficAssignment, TrafficClass
from aequilibrae.project.network.visum_geojson_importer import (
    CONNECTOR_FALLBACK_CAPACITY,
    CONNECTOR_FALLBACK_SPEED_KMH,
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
            "link_id": 1,
            "counts": {"DTVW": 1300, "HVG_ORIG": 120, "MOTOR_ORIG": 1070, "CAR_ORIG": 950},
        }
    ]
    assert any(diag.code == "deferred-count-fields" for diag in report.diagnostics)

    with empty_project.db_connection as conn:
        assert conn.execute("select count(*) from nodes").fetchone()[0] == 4
        assert conn.execute("select count(*) from links").fetchone()[0] == 4
        assert conn.execute("select count(*) from zones").fetchone()[0] == 2
        assert conn.execute("select count(*) from modes where mode_id='h'").fetchone()[0] == 1
        assert conn.execute("select link_id from links order by link_id").fetchall() == [(1,), (2,), (3,), (4,)]
        assert conn.execute("select visum_link_no from links where link_id=1").fetchone()[0] == 100
        assert conn.execute("select direction, modes from links where link_id=1").fetchone() == (1, "ch")
        assert conn.execute("select direction, modes from links where link_id=2").fetchone() == (-1, "c")
        assert conn.execute("select visum_length_ab from links where visum_link_no=100").fetchone()[0] == 1000
        assert conn.execute("select a_node, b_node from links where visum_link_no=100").fetchone() == (1, 2)
        assert conn.execute("select distance > 0 from links where visum_link_no=100").fetchone()[0] == 1
        assert conn.execute("select modes is not null from nodes where node_id=1").fetchone()[0] == 1
        assert conn.execute("select link_types is not null from nodes where node_id=1").fetchone()[0] == 1
        assert conn.execute("select a_node, b_node from links where visum_connector_no=9001").fetchone() == (1001, 1)
        assert conn.execute("select modes from links where visum_connector_no=9001").fetchone()[0] == "ch"
        assert (
            conn.execute("select visum_connector_key from links where visum_connector_no=9001").fetchone()[0]
            == "connector:1001:1:B"
        )
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


def test_coincident_nodes_are_offset_to_preserve_topology(empty_project, visum_geojson_folder):
    nodes = gpd.read_file(visum_geojson_folder / "node.geojson")
    nodes.loc[len(nodes)] = {"NO": 3, "NAME": "C", "geometry": Point(0.01, 0.0)}
    nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs="EPSG:4326")
    nodes.to_file(visum_geojson_folder / "node.geojson", driver="GeoJSON")

    links = gpd.read_file(visum_geojson_folder / "link.geojson")
    duplicate_link = links.iloc[0].copy()
    duplicate_link["NO"] = 101
    duplicate_link["FROMNODENO"] = 3
    duplicate_link["TONODENO"] = 1
    duplicate_link["geometry"] = LineString([(0.01, 0.0, 5.0), (0.005, 0.0, 5.0), (0.0, 0.0, 5.0)])
    links = gpd.GeoDataFrame([links.iloc[0], duplicate_link], geometry="geometry", crs="EPSG:4326")
    links.to_file(visum_geojson_folder / "link.geojson", driver="GeoJSON")

    report = empty_project.network.create_from_visum_geojson(visum_geojson_folder)

    assert report.imported_counts["nodes"] == 3
    assert report.imported_counts["links"] == 2
    assert any(diag.code == "coincident-node-offset" for diag in report.diagnostics)
    assert report.source_references["node_coordinate_offsets"][2]["offset_m"] == 0.0
    assert report.source_references["node_coordinate_offsets"][3]["offset_m"] == 0.25

    with empty_project.db_connection_spatial as conn:
        rows = conn.execute(
            """
            SELECT node_id, ST_X(geometry), ST_Y(geometry), visum_original_lon, visum_original_lat,
                   visum_duplicate_coord_group, visum_coord_offset_m
            FROM nodes
            WHERE node_id in (2, 3)
            ORDER BY node_id
            """
        ).fetchall()
        link_endpoint_matches_node = conn.execute(
            """
            SELECT Abs(ST_X(StartPoint(links.geometry)) - ST_X(nodes.geometry)) < 1e-12
                   AND Abs(ST_Y(StartPoint(links.geometry)) - ST_Y(nodes.geometry)) < 1e-12
            FROM links
            JOIN nodes ON links.a_node = nodes.node_id
            WHERE links.visum_link_no = 101
            """
        ).fetchone()[0]

    assert rows[0][1:5] == (0.01, 0.0, 0.01, 0.0)
    assert rows[0][5] == rows[1][5] == "node-coordinate-1"
    assert rows[0][6] == 0.0
    assert rows[1][3:5] == (0.01, 0.0)
    assert rows[1][1] != rows[1][3] or rows[1][2] != rows[1][4]
    assert rows[1][6] == 0.25
    assert link_endpoint_matches_node == 1


def test_sparse_visum_link_numbers_import_as_compact_link_ids(empty_project, visum_geojson_folder):
    links = gpd.read_file(visum_geojson_folder / "link.geojson")
    links.loc[links.index[0], "NO"] = 2_000_001_598
    links.to_file(visum_geojson_folder / "link.geojson", driver="GeoJSON")

    report = empty_project.network.create_from_visum_geojson(visum_geojson_folder)

    assert report.source_references["links"][2_000_001_598] == 1
    assert report.source_references["count_locations"] == []
    assert any(diag.code == "unresolved-count-link" for diag in report.diagnostics)

    with empty_project.db_connection as conn:
        assert conn.execute("select min(link_id), max(link_id), count(*) from links").fetchone() == (1, 4, 4)
        assert conn.execute("select visum_link_no from links where link_id=1").fetchone()[0] == 2_000_001_598


def test_coincident_node_error_policy_rejects_before_import(empty_project, visum_geojson_folder):
    nodes = gpd.read_file(visum_geojson_folder / "node.geojson")
    nodes.loc[len(nodes)] = {"NO": 3, "NAME": "C", "geometry": Point(0.01, 0.0)}
    nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs="EPSG:4326")
    nodes.to_file(visum_geojson_folder / "node.geojson", driver="GeoJSON")

    with pytest.raises(ValueError, match="coincident-node-coordinate"):
        empty_project.network.create_from_visum_geojson(visum_geojson_folder, duplicate_node_policy="error")

    with empty_project.db_connection as conn:
        assert conn.execute("select count(*) from nodes").fetchone()[0] == 0
        assert conn.execute("select count(*) from links").fetchone()[0] == 0


def test_source_node_id_collision_with_zone_id_is_remapped(empty_project, visum_geojson_folder):
    nodes = gpd.read_file(visum_geojson_folder / "node.geojson")
    nodes.loc[len(nodes)] = {"NO": 1001, "NAME": "source node with zone id", "geometry": Point(0.03, 0.0)}
    nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs="EPSG:4326")
    nodes.to_file(visum_geojson_folder / "node.geojson", driver="GeoJSON")

    links = gpd.read_file(visum_geojson_folder / "link.geojson")
    remapped_link = links.iloc[0].copy()
    remapped_link["NO"] = 101
    remapped_link["FROMNODENO"] = 1001
    remapped_link["TONODENO"] = 1
    remapped_link["geometry"] = LineString([(0.03, 0.0), (0.0, 0.0)])
    links = gpd.GeoDataFrame([links.iloc[0], remapped_link], geometry="geometry", crs="EPSG:4326")
    links.to_file(visum_geojson_folder / "link.geojson", driver="GeoJSON")

    report = empty_project.network.create_from_visum_geojson(visum_geojson_folder)
    remapped_node_id = report.source_references["nodes"][1001]

    assert remapped_node_id != 1001
    assert any(diag.code == "node-id-remapped" and diag.source_id == 1001 for diag in report.diagnostics)

    with empty_project.db_connection as conn:
        assert conn.execute("select is_centroid from nodes where node_id=1001").fetchone()[0] == 1
        source_node_no = conn.execute(
            "select visum_node_no from nodes where node_id=?", (remapped_node_id,)
        ).fetchone()[0]
        assert conn.execute("select a_node, b_node from links where visum_link_no=101").fetchone() == (
            remapped_node_id,
            1,
        )

    assert source_node_no == 1001


def test_coincident_zone_centroid_is_offset_and_connectors_follow(empty_project, visum_geojson_folder):
    centroids = gpd.read_file(visum_geojson_folder / "zone_centroid.geojson")
    centroids.loc[centroids["NO"] == 1001, "geometry"] = Point(0.0, 0.0)
    centroids.to_file(visum_geojson_folder / "zone_centroid.geojson", driver="GeoJSON")

    connectors = gpd.read_file(visum_geojson_folder / "connector.geojson")
    connectors.loc[connectors["ZONENO"] == 1001, "geometry"] = LineString([(0.0, 0.0), (0.0, 0.0)])
    connectors.to_file(visum_geojson_folder / "connector.geojson", driver="GeoJSON")

    report = empty_project.network.create_from_visum_geojson(visum_geojson_folder)

    assert any(diag.code == "coincident-centroid-offset" and diag.source_id == 1001 for diag in report.diagnostics)
    assert report.source_references["zone_coordinate_offsets"][1001]["offset_m"] == 0.25

    with empty_project.db_connection_spatial as conn:
        centroid = conn.execute(
            """
            SELECT ST_X(geometry), ST_Y(geometry), visum_original_lon, visum_original_lat,
                   visum_duplicate_coord_group, visum_coord_offset_m
            FROM nodes
            WHERE node_id = 1001
            """
        ).fetchone()
        connector_start_matches_centroid = conn.execute(
            """
            SELECT Abs(ST_X(StartPoint(links.geometry)) - ST_X(nodes.geometry)) < 1e-12
                   AND Abs(ST_Y(StartPoint(links.geometry)) - ST_Y(nodes.geometry)) < 1e-12
            FROM links
            JOIN nodes ON links.a_node = nodes.node_id
            WHERE links.visum_connector_no = 9001
            """
        ).fetchone()[0]

    assert centroid[2:4] == (0.0, 0.0)
    assert centroid[0] != centroid[2] or centroid[1] != centroid[3]
    assert centroid[4] == "zone-coordinate-1"
    assert centroid[5] == 0.25
    assert connector_start_matches_centroid == 1


def test_connector_without_no_gets_deterministic_source_key(empty_project, visum_geojson_folder):
    connectors = gpd.read_file(visum_geojson_folder / "connector.geojson").drop(columns=["NO"])
    connectors.to_file(visum_geojson_folder / "connector.geojson", driver="GeoJSON")

    report = empty_project.network.create_from_visum_geojson(visum_geojson_folder)

    assert report.source_references["connectors"] == {
        "connector:1001:1:B": 3,
        "connector:1002:2:B": 4,
    }
    with empty_project.db_connection as conn:
        assert conn.execute("select visum_connector_no from links where link_id=3").fetchone()[0] is None
        assert (
            conn.execute("select visum_connector_key from links where link_id=3").fetchone()[0]
            == "connector:1001:1:B"
        )


def test_duplicate_connector_source_keys_get_stable_suffix(empty_project, visum_geojson_folder):
    connectors = gpd.read_file(visum_geojson_folder / "connector.geojson").drop(columns=["NO"])
    connectors.loc[len(connectors)] = connectors.iloc[0]
    connectors = gpd.GeoDataFrame(connectors, geometry="geometry", crs="EPSG:4326")
    connectors.to_file(visum_geojson_folder / "connector.geojson", driver="GeoJSON")

    report = empty_project.network.create_from_visum_geojson(visum_geojson_folder)

    assert report.source_references["connectors"] == {
        "connector:1001:1:B": 3,
        "connector:1002:2:B": 4,
        "connector:1001:1:B:2": 5,
    }
    with empty_project.db_connection as conn:
        keys = conn.execute("select visum_connector_key from links where link_type='centroid_connector'").fetchall()

    assert [row[0] for row in keys] == ["connector:1001:1:B", "connector:1002:2:B", "connector:1001:1:B:2"]


def _external_visum_geojson_folder(request):
    folder = request.config.getoption("--visum-geojson-folder")
    if folder is None:
        pytest.skip("Pass --visum-geojson-folder to run the external VISUM GeoJSON import smoke test")

    folder = Path(folder)
    if not folder.exists():
        pytest.fail(f"VISUM GeoJSON folder does not exist: {folder}")
    if not folder.is_dir():
        pytest.fail(f"VISUM GeoJSON input must be a folder: {folder}")
    return folder


def _expected_external_visum_counts(request, folder):
    option_names = {
        "nodes": "--visum-expected-nodes",
        "zones": "--visum-expected-zones",
        "links": "--visum-expected-links",
        "connectors": "--visum-expected-connectors",
    }
    layer_names = {
        "nodes": "node",
        "zones": "zone_centroid",
        "links": "link",
        "connectors": "connector",
    }
    report = discover_visum_geojson_layers(folder)
    report.raise_for_errors()

    expected = {}
    for count_name, option_name in option_names.items():
        option_value = request.config.getoption(option_name)
        if option_value is not None:
            expected[count_name] = option_value
            continue

        layer_path = report.discovered_layers[layer_names[count_name]]
        expected[count_name] = len(gpd.read_file(layer_path))

    return expected


def test_external_visum_geojson_imports_expected_counts(empty_project, request):
    folder = _external_visum_geojson_folder(request)
    expected_counts = _expected_external_visum_counts(request, folder)

    report = empty_project.network.create_from_visum_geojson(folder)

    assert report.imported_counts == expected_counts
    with empty_project.db_connection as conn:
        assert conn.execute("select count(*) from nodes").fetchone()[0] == (
            expected_counts["nodes"] + expected_counts["zones"]
        )
        assert conn.execute("select count(*) from zones").fetchone()[0] == expected_counts["zones"]
        assert conn.execute("select count(*) from links").fetchone()[0] == (
            expected_counts["links"] + expected_counts["connectors"]
        )


def test_link_type_override(empty_project, visum_geojson_folder):
    empty_project.network.create_from_visum_geojson(visum_geojson_folder, link_type_mapping={"ARTERIAL": "arterial"})

    with empty_project.db_connection as conn:
        assert conn.execute("select link_type from links where visum_link_no=100").fetchone()[0] == "arterial"


def test_numeric_typeno_link_types_get_distinct_generated_names(empty_project, visum_geojson_folder):
    link = gpd.read_file(visum_geojson_folder / "link.geojson")
    second = link.iloc[0].copy()
    second["NO"] = 101
    second["TYPENO"] = 92
    second["R_TYPENO"] = 92
    link = gpd.GeoDataFrame([link.iloc[0], second], geometry="geometry", crs="EPSG:4326").reset_index(drop=True)
    link = link.drop(columns=["LC", "R_LC"])
    link.loc[0, "TYPENO"] = 2
    link.loc[0, "R_TYPENO"] = 2
    link.to_file(visum_geojson_folder / "link.geojson", driver="GeoJSON")

    report = empty_project.network.create_from_visum_geojson(visum_geojson_folder)

    assert report.link_type_mapping == {"2": "visum_two", "92": "visum_nine_two"}
    with empty_project.db_connection as conn:
        assert conn.execute("select link_type from links where visum_link_no=100").fetchone()[0] == "visum_two"
        assert conn.execute("select link_type from links where visum_link_no=101").fetchone()[0] == "visum_nine_two"


def test_mode_override_merges_hgv_into_car(empty_project, visum_geojson_folder):
    report = empty_project.network.create_from_visum_geojson(
        visum_geojson_folder, mode_mapping={"CAR": "c", "HGV": "c"}
    )

    with empty_project.db_connection as conn:
        modes = conn.execute("select modes from links where visum_link_no=100").fetchone()[0]
        h_count = conn.execute("select count(*) from modes where mode_id='h'").fetchone()[0]

    assert modes == "c"
    assert h_count == 0
    assert report.mode_mapping == {"CAR": "c", "HGV": "c"}


def test_extra_transport_system_requires_mapping_or_ignore(empty_project, visum_geojson_folder):
    link = gpd.read_file(visum_geojson_folder / "link.geojson")
    link.loc[0, "TSYSSET"] = "CAR,HGV,BUS"
    link.to_file(visum_geojson_folder / "link.geojson", driver="GeoJSON")

    with pytest.raises(ValueError, match="BUS"):
        empty_project.network.create_from_visum_geojson(visum_geojson_folder)

    with empty_project.db_connection as conn:
        assert conn.execute("select count(*) from links").fetchone()[0] == 0


def test_ignored_transport_system_is_reported_and_not_imported(empty_project, visum_geojson_folder):
    link = gpd.read_file(visum_geojson_folder / "link.geojson")
    link.loc[0, "TSYSSET"] = "CAR,HGV,BUS"
    link.to_file(visum_geojson_folder / "link.geojson", driver="GeoJSON")

    report = empty_project.network.create_from_visum_geojson(
        visum_geojson_folder, ignored_transport_systems={"BUS"}
    )

    assert any(diag.code == "ignored-transport-system" and "BUS" in diag.message for diag in report.diagnostics)
    with empty_project.db_connection as conn:
        assert conn.execute("select modes from links where visum_link_no=100").fetchone()[0] == "ch"


def test_records_with_only_ignored_transport_systems_are_skipped(empty_project, visum_geojson_folder):
    link = gpd.read_file(visum_geojson_folder / "link.geojson")
    link.loc[0, "TSYSSET"] = "BUS"
    link.loc[0, "R_TSYSSET"] = "BUS"
    link.to_file(visum_geojson_folder / "link.geojson", driver="GeoJSON")

    report = empty_project.network.create_from_visum_geojson(
        visum_geojson_folder, ignored_transport_systems={"BUS"}
    )

    assert report.imported_counts["links"] == 0
    assert report.imported_counts["connectors"] == 2
    assert any(diag.code == "ignored-record" and diag.layer == "link" for diag in report.diagnostics)

    with empty_project.db_connection as conn:
        assert conn.execute("select count(*) from links where visum_link_no=100").fetchone()[0] == 0
        assert conn.execute("select count(*) from links where link_type='centroid_connector'").fetchone()[0] == 2


def test_records_with_empty_transport_systems_are_skipped(empty_project, visum_geojson_folder):
    link = gpd.read_file(visum_geojson_folder / "link.geojson")
    link.loc[0, "TSYSSET"] = ""
    link.loc[0, "R_TSYSSET"] = None
    link.to_file(visum_geojson_folder / "link.geojson", driver="GeoJSON")

    report = empty_project.network.create_from_visum_geojson(visum_geojson_folder)

    assert report.imported_counts["links"] == 0
    assert any(diag.code == "empty-transport-systems" and diag.layer == "link" for diag in report.diagnostics)

    with empty_project.db_connection as conn:
        assert conn.execute("select count(*) from links where visum_link_no=100").fetchone()[0] == 0


def test_user_can_map_bus_as_assignable_mode(empty_project, visum_geojson_folder):
    link = gpd.read_file(visum_geojson_folder / "link.geojson")
    link.loc[0, "TSYSSET"] = "CAR,HGV,BUS"
    link.loc[0, "R_TSYSSET"] = "CAR,BUS"
    link.to_file(visum_geojson_folder / "link.geojson", driver="GeoJSON")

    report = empty_project.network.create_from_visum_geojson(
        visum_geojson_folder, mode_mapping={"CAR": "c", "HGV": "h", "BUS": "t"}
    )

    assert report.mode_mapping == {"CAR": "c", "HGV": "h", "BUS": "t"}
    with empty_project.db_connection as conn:
        assert conn.execute("select modes from links where visum_link_no=100").fetchone()[0] == "cht"
        assert conn.execute("select count(*) from modes where mode_id='t'").fetchone()[0] == 1


def test_mode_excluded_missing_fields_do_not_poison_car_graph(empty_project, visum_geojson_folder):
    links = gpd.read_file(visum_geojson_folder / "link.geojson")
    transit_only = links.iloc[0].copy()
    transit_only["NO"] = 200
    transit_only["TSYSSET"] = "BUS"
    transit_only["R_TSYSSET"] = "BUS"
    for field in ("V0PRT", "R_V0PRT", "CAPPRT", "R_CAPPRT", "T0PRT", "R_T0PRT"):
        if field in transit_only.index:
            transit_only[field] = None
    links = gpd.GeoDataFrame([links.iloc[0], transit_only], geometry="geometry", crs="EPSG:4326")
    links.to_file(visum_geojson_folder / "link.geojson", driver="GeoJSON")

    empty_project.network.create_from_visum_geojson(
        visum_geojson_folder, mode_mapping={"CAR": "c", "HGV": "h", "BUS": "t"}
    )
    with empty_project.db_connection as conn:
        transit_link_id = conn.execute("select link_id from links where visum_link_no=200").fetchone()[0]
    empty_project.network.build_graphs(
        fields=["distance", "travel_time_ab", "travel_time_ba", "capacity_ab", "capacity_ba"], modes=["c"]
    )

    graph = empty_project.network.graphs["c"].graph
    transit_self_loop = graph[graph.link_id == transit_link_id]

    assert transit_self_loop.a_node.eq(transit_self_loop.b_node).all()
    assert not graph.travel_time.isna().any()
    assert not graph.capacity.isna().any()


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


def test_connector_assignment_fields_default_when_not_exported(visum_geojson_folder, empty_project):
    connectors = gpd.read_file(visum_geojson_folder / "connector.geojson")
    connectors = connectors.drop(columns=["V0PRT", "R_V0PRT", "CAPPRT", "R_CAPPRT"])
    connectors.loc[connectors.index[0], ["LENGTH", "R_LENGTH"]] = "0km"
    connectors.to_file(visum_geojson_folder / "connector.geojson", driver="GeoJSON")

    report = empty_project.network.create_from_visum_geojson(visum_geojson_folder)

    expected_time = (100.0 / 1000.0) / CONNECTOR_FALLBACK_SPEED_KMH * 60.0

    assert any(diag.code == "connector-length-defaulted" for diag in report.diagnostics)
    assert any(diag.code == "connector-speed-defaulted" for diag in report.diagnostics)
    assert any(diag.code == "connector-capacity-defaulted" for diag in report.diagnostics)

    with empty_project.db_connection as conn:
        row = conn.execute(
            """
            SELECT travel_time_ab, travel_time_ba, capacity_ab, capacity_ba
            FROM links
            WHERE visum_connector_no=9001
            """
        ).fetchone()

    assert row[0] > expected_time
    assert row[1] > expected_time
    assert row[2:] == (CONNECTOR_FALLBACK_CAPACITY, CONNECTOR_FALLBACK_CAPACITY)
