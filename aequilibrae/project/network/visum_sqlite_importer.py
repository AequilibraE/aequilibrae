import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import geopandas as gpd
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from aequilibrae.project.network.visum_geojson_importer import (
    CONNECTOR_FALLBACK_CAPACITY,
    DEFAULT_MODE_MAPPING,
    VisumGeoJSONImporter,
    VisumGeoJSONReport,
    _direction,
    _jsonish,
    _split_tsysset,
)


REQUIRED_SQLITE_TABLES = {"NETWORK", "NODE", "LINK", "CONNECTOR", "ZONE", "TSYS", "LINKTYPE"}
OPTIONAL_SQLITE_TABLES = {
    "MODE",
    "COUNTLOCATION",
    "LINKPOLY",
    "SURFACEITEM",
    "FACEITEM",
    "EDGE",
    "EDGEITEM",
    "POINT",
}
DEFERRED_SQLITE_TABLES = {
    "STOP",
    "STOPPOINT",
    "STOPAREA",
    "LINE",
    "LINEROUTE",
    "LINEROUTEITEM",
    "TIMEPROFILE",
    "TIMEPROFILEITEM",
    "VEHJOURNEY",
    "VEHJOURNEYITEM",
    "TURN",
    "LANETURN",
    "FARESYSTEM",
    "FAREMODEL",
}
VISUM_SQLITE_CONNECTOR_EPSILON_MINUTES = 1e-6


@dataclass
class VisumSQLiteReport(VisumGeoJSONReport):
    """Diagnostics and provenance returned by a VISUM SQLite import."""

    def raise_for_errors(self) -> None:
        if self.errors:
            messages = "; ".join(f"{diag.code}: {diag.message}" for diag in self.errors[:5])
            raise ValueError(f"VISUM SQLite import failed validation: {messages}")


def discover_visum_sqlite(path: str | Path) -> VisumSQLiteReport:
    """Validate a VISUM SQLite export and report available required, optional, and deferred tables."""

    report = VisumSQLiteReport()
    db_path = Path(path)
    if not db_path.exists():
        report.add("error", "missing-input", f"VISUM SQLite path does not exist: {db_path}")
        return report
    if not db_path.is_file():
        report.add("error", "invalid-input", f"VISUM SQLite input must be a file: {db_path}")
        return report

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')").fetchall()
    except sqlite3.Error as exc:
        report.add("error", "invalid-sqlite", f"Could not read VISUM SQLite database: {exc}")
        return report

    tables = {row[0].upper(): row[0] for row in rows}
    table_names = set(tables)
    report.discovered_layers.update(
        {table.lower(): tables[table] for table in sorted(table_names & REQUIRED_SQLITE_TABLES)}
    )
    report.source_references["sqlite_tables"] = sorted(tables)

    for table in sorted(REQUIRED_SQLITE_TABLES - table_names):
        report.add("error", "missing-table", f"Required VISUM SQLite table '{table}' was not provided", layer=table)

    for table in sorted(table_names & DEFERRED_SQLITE_TABLES):
        report.deferred_layers.append(table)
        report.add(
            "warning",
            "deferred-table",
            f"VISUM SQLite table '{table}' is recognized but deferred",
            layer=table,
        )
    return report


def visum_sqlite_source_connectivity(
    path: str | Path,
    *,
    mode_mapping: Mapping[str, str] | None = None,
    ignored_transport_systems: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, set[tuple[int, int]]]:
    """Extract source directed link/connector connectivity by mapped AequilibraE mode."""

    mapping = {str(k).upper(): v for k, v in (mode_mapping or DEFAULT_MODE_MAPPING).items()}
    ignored = {str(token).upper() for token in (ignored_transport_systems or set())}
    connectivity = {mode: set() for mode in sorted(set(mapping.values()))}

    with sqlite3.connect(Path(path)) as conn:
        for row in conn.execute('SELECT FROMNODENO, TONODENO, TSYSSET FROM "LINK"'):
            from_node, to_node, tsysset = row
            for mode in _mapped_modes_from_value(tsysset, mapping, ignored):
                connectivity.setdefault(mode, set()).add((int(from_node), int(to_node)))

        for zone_no, node_no, direction, tsysset in conn.execute(
            'SELECT ZONENO, NODENO, DIRECTION, TSYSSET FROM "CONNECTOR"'
        ):
            if direction == "O":
                arc = (int(zone_no), int(node_no))
            elif direction == "D":
                arc = (int(node_no), int(zone_no))
            else:
                continue
            for mode in _mapped_modes_from_value(tsysset, mapping, ignored):
                connectivity.setdefault(mode, set()).add(arc)
    return connectivity


class VisumSQLiteImporter(VisumGeoJSONImporter):
    def __init__(
        self,
        net,
        path: str | Path,
        *,
        mode_mapping: Mapping[str, str] | None = None,
        ignored_transport_systems: set[str] | list[str] | tuple[str, ...] | None = None,
        link_type_mapping: Mapping[object, str] | None = None,
        source_crs: str | int | None = None,
        accept_default_crs: bool = False,
        default_crs: str = "EPSG:4326",
        allow_non_empty: bool = False,
        geometry_tolerance: float = 1e-6,
        duplicate_node_policy: str = "offset",
        duplicate_node_offset_meters: float = 0.25,
        connector_epsilon_minutes: float = VISUM_SQLITE_CONNECTOR_EPSILON_MINUTES,
        connector_capacity: float = CONNECTOR_FALLBACK_CAPACITY,
    ) -> None:
        super().__init__(
            net,
            path,
            mode_mapping=mode_mapping,
            ignored_transport_systems=ignored_transport_systems,
            link_type_mapping=link_type_mapping,
            source_crs=source_crs,
            accept_default_crs=accept_default_crs,
            allow_non_empty=allow_non_empty,
            geometry_tolerance=geometry_tolerance,
            duplicate_node_policy=duplicate_node_policy,
            duplicate_node_offset_meters=duplicate_node_offset_meters,
        )
        if connector_epsilon_minutes <= 0:
            raise ValueError("connector_epsilon_minutes must be positive")
        self.path = Path(path)
        self.default_crs = default_crs
        self.connector_epsilon_minutes = connector_epsilon_minutes
        self.connector_capacity = connector_capacity
        self.report = VisumSQLiteReport(mode_mapping=dict(self.mode_mapping))

    def doWork(self) -> VisumSQLiteReport:
        report = super().doWork()
        report.add("info", "import-complete", "VISUM SQLite import completed")
        return report

    def _discover_and_read(self) -> None:
        discovery = discover_visum_sqlite(self.path)
        self.report.discovered_layers.update(discovery.discovered_layers)
        self.report.deferred_layers.extend(discovery.deferred_layers)
        self.report.diagnostics.extend(discovery.diagnostics)
        self.report.source_references.update(discovery.source_references)
        self.report.raise_for_errors()

        try:
            self.layers = _read_sqlite_as_visum_layers(
                self.path,
                self.report,
                mode_mapping=self.mode_mapping,
                ignored_transport_systems=self.ignored_transport_systems,
                source_crs=self.source_crs,
                accept_default_crs=self.accept_default_crs,
                default_crs=self.default_crs,
                connector_epsilon_minutes=self.connector_epsilon_minutes,
                connector_capacity=self.connector_capacity,
            )
        except sqlite3.Error as exc:
            self.report.add("error", "sqlite-read-failed", f"Could not read VISUM SQLite tables: {exc}")
        self.report.field_inventory = _inventory_sqlite_layers(self.layers)

    def _validate_required_columns(self) -> None:
        super()._validate_required_columns()
        required = {
            "network": {"PROJECTIONDEFINITION"},
            "tsys": {"CODE", "TYPE"},
            "linktype": {"NO"},
        }
        for layer, fields in required.items():
            if layer not in self.layers:
                continue
            missing = fields - set(self.layers[layer].columns)
            for field_name in sorted(missing):
                self.report.add(
                    "error",
                    "missing-field",
                    f"Table-derived layer '{layer}' is missing required field '{field_name}'",
                    layer=layer,
                    field=field_name,
                )

    def _prepare_connector_source_keys(self) -> None:
        if "connector" not in self.layers:
            return
        for row_index, row in self.layers["connector"].iterrows():
            ab_modes = self._mapped_modes(row.get("TSYSSET"), "connector", None)
            ba_modes = self._mapped_modes(row.get("R_TSYSSET"), "connector", None)
            direction_code = "B" if _direction(ab_modes, ba_modes) == 0 else ("O" if ab_modes else "D")
            self.connector_source_keys[row_index] = (
                f"connector:{int(row['ZONENO'])}:{int(row['NODENO'])}:{direction_code}"
            )
            self.connector_source_nos[row_index] = None


def _read_sqlite_as_visum_layers(
    path: Path,
    report: VisumSQLiteReport,
    *,
    mode_mapping: Mapping[str, str],
    ignored_transport_systems: set[str],
    source_crs: str | int | None,
    accept_default_crs: bool,
    default_crs: str,
    connector_epsilon_minutes: float,
    connector_capacity: float,
) -> dict[str, gpd.GeoDataFrame]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        network = _read_table(conn, "NETWORK")
        transformer = _coordinate_transformer(network, report, source_crs, accept_default_crs, default_crs)
        nodes = _read_table(conn, "NODE")
        zones = _read_table(conn, "ZONE")
        links = _read_table(conn, "LINK")
        connectors = _read_table(conn, "CONNECTOR")
        tsys = _read_table(conn, "TSYS")
        link_types = _read_table(conn, "LINKTYPE")
        modes = _read_optional_table(conn, "MODE")
        count_locations = _read_optional_table(conn, "COUNTLOCATION")

        point_by_node = _point_lookup(nodes, transformer)
        point_by_zone = _point_lookup(zones, transformer, id_column="NO")
        linkpolys = _read_linkpolys(conn, transformer)
        zone_polygons = _read_zone_polygons(conn, zones, transformer, report)

    node_layer = _node_layer(nodes, point_by_node)
    zone_layer = _zone_layer(zones, point_by_zone)
    link_layer = _link_layer(links, point_by_node, linkpolys, report)
    connector_layer = _connector_layer(
        connectors,
        point_by_zone,
        point_by_node,
        report,
        mode_mapping,
        ignored_transport_systems,
        connector_epsilon_minutes,
        connector_capacity,
    )
    layers: dict[str, gpd.GeoDataFrame] = {
        "network": gpd.GeoDataFrame(network),
        "tsys": gpd.GeoDataFrame(tsys),
        "linktype": gpd.GeoDataFrame(link_types),
        "node": node_layer,
        "zone_centroid": zone_layer,
        "link": link_layer,
        "connector": connector_layer,
    }
    if modes is not None:
        layers["mode"] = gpd.GeoDataFrame(modes)
    if count_locations is not None:
        layers["countlocation"] = _count_location_layer(count_locations, link_layer)
    if zone_polygons:
        layers["zone_polygon"] = gpd.GeoDataFrame(list(zone_polygons.values()), geometry="geometry", crs="EPSG:4326")
    return layers


def _read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)


def _read_optional_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame | None:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND UPPER(name)=?",
        (table.upper(),),
    ).fetchone()
    if exists is None:
        return None
    return _read_table(conn, table)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND UPPER(name)=?",
            (table.upper(),),
        ).fetchone()
        is not None
    )


def _coordinate_transformer(
    network: pd.DataFrame,
    report: VisumSQLiteReport,
    source_crs: str | int | None,
    accept_default_crs: bool,
    default_crs: str,
) -> Transformer:
    crs_value = source_crs
    if crs_value is None and "PROJECTIONDEFINITION" in network.columns and not network.empty:
        crs_value = network.iloc[0].get("PROJECTIONDEFINITION")
    if not crs_value:
        if not accept_default_crs:
            report.add(
                "error",
                "missing-crs",
                "VISUM SQLite source CRS is missing; supply source_crs or accept_default_crs=True",
            )
            report.raise_for_errors()
        crs_value = default_crs
        report.add("warning", "default-crs-assumed", f"Assuming VISUM SQLite source CRS {default_crs}")
    try:
        crs = CRS.from_user_input(crs_value)
    except Exception as exc:  # pyproj raises multiple exception types depending on the CRS payload
        report.add("error", "invalid-crs", f"Could not parse VISUM SQLite source CRS: {exc}")
        report.raise_for_errors()
        crs = CRS.from_user_input(default_crs)
    report.crs["sqlite"] = f"{crs.to_string()} -> EPSG:4326"
    return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def _transform_point(transformer: Transformer, x, y) -> Point:
    lon, lat = transformer.transform(float(x), float(y))
    return Point(lon, lat)


def _point_lookup(df: pd.DataFrame, transformer: Transformer, *, id_column: str = "NO") -> dict[int, Point]:
    return {
        int(row[id_column]): _transform_point(transformer, row["XCOORD"], row["YCOORD"])
        for _, row in df.iterrows()
        if pd.notna(row.get("XCOORD")) and pd.notna(row.get("YCOORD"))
    }


def _node_layer(nodes: pd.DataFrame, point_by_node: Mapping[int, Point]) -> gpd.GeoDataFrame:
    rows = []
    for _, row in nodes.iterrows():
        source_id = int(row["NO"])
        if source_id not in point_by_node:
            continue
        rows.append({**row.to_dict(), "NO": source_id, "geometry": point_by_node[source_id]})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _zone_layer(zones: pd.DataFrame, point_by_zone: Mapping[int, Point]) -> gpd.GeoDataFrame:
    rows = []
    for _, row in zones.iterrows():
        zone_id = int(row["NO"])
        if zone_id not in point_by_zone:
            continue
        rows.append({**row.to_dict(), "NO": zone_id, "geometry": point_by_zone[zone_id]})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _read_linkpolys(conn: sqlite3.Connection, transformer: Transformer) -> dict[tuple[int, int], list[Point]]:
    if not _table_exists(conn, "LINKPOLY"):
        return {}
    rows = conn.execute(
        'SELECT FROMNODENO, TONODENO, "INDEX", XCOORD, YCOORD FROM "LINKPOLY" ORDER BY FROMNODENO, TONODENO, "INDEX"'
    ).fetchall()
    linkpolys: dict[tuple[int, int], list[Point]] = {}
    for row in rows:
        key = (int(row["FROMNODENO"]), int(row["TONODENO"]))
        linkpolys.setdefault(key, []).append(_transform_point(transformer, row["XCOORD"], row["YCOORD"]))
    return linkpolys


def _link_layer(
    links: pd.DataFrame,
    point_by_node: Mapping[int, Point],
    linkpolys: Mapping[tuple[int, int], list[Point]],
    report: VisumSQLiteReport,
) -> gpd.GeoDataFrame:
    rows = []
    for source_no, group in links.groupby("NO", sort=True):
        group = group.sort_values(["FROMNODENO", "TONODENO"])
        ab = group.iloc[0]
        reverse = group[
            (group["FROMNODENO"] == ab["TONODENO"])
            & (group["TONODENO"] == ab["FROMNODENO"])
            & (group.index != ab.name)
        ]
        ba = reverse.iloc[0] if not reverse.empty else None
        from_node = int(ab["FROMNODENO"])
        to_node = int(ab["TONODENO"])
        if from_node not in point_by_node or to_node not in point_by_node:
            report.add(
                "error",
                "missing-node-reference",
                "SQLite link references missing node coordinates",
                layer="link",
                source_id=int(source_no),
            )
            continue
        geometry = _link_geometry(from_node, to_node, point_by_node, linkpolys)
        row = _directional_link_row(source_no, ab, ba)
        row["geometry"] = geometry
        rows.append(row)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _directional_link_row(source_no, ab: pd.Series, ba: pd.Series | None) -> dict:
    row = {
        "NO": int(source_no),
        "FROMNODENO": int(ab["FROMNODENO"]),
        "TONODENO": int(ab["TONODENO"]),
        "TSYSSET": _string_or_empty(ab.get("TSYSSET")),
        "LC": _string_or_empty(ab.get("LC")),
        "TYPENO": ab.get("TYPENO"),
        "LENGTH": _sqlite_length(ab.get("LENGTH")),
        "V0PRT": _sqlite_speed(ab.get("V0PRT")),
        "CAPPRT": _sqlite_capacity(ab.get("CAPPRT")),
    }
    if ba is not None:
        row.update(
            {
                "R_TSYSSET": _string_or_empty(ba.get("TSYSSET")),
                "R_LC": _string_or_empty(ba.get("LC")),
                "R_TYPENO": ba.get("TYPENO"),
                "R_LENGTH": _sqlite_length(ba.get("LENGTH")),
                "R_V0PRT": _sqlite_speed(ba.get("V0PRT")),
                "R_CAPPRT": _sqlite_capacity(ba.get("CAPPRT")),
            }
        )
    else:
        row.update({"R_TSYSSET": "", "R_LC": "", "R_TYPENO": None, "R_LENGTH": None, "R_V0PRT": None, "R_CAPPRT": None})
    return row


def _link_geometry(
    from_node: int,
    to_node: int,
    point_by_node: Mapping[int, Point],
    linkpolys: Mapping[tuple[int, int], list[Point]],
) -> LineString:
    if (from_node, to_node) in linkpolys:
        vertices = linkpolys[(from_node, to_node)]
    elif (to_node, from_node) in linkpolys:
        vertices = list(reversed(linkpolys[(to_node, from_node)]))
    else:
        vertices = []
    coords = [point_by_node[from_node], *vertices, point_by_node[to_node]]
    return LineString([(point.x, point.y) for point in coords])


def _connector_layer(
    connectors: pd.DataFrame,
    point_by_zone: Mapping[int, Point],
    point_by_node: Mapping[int, Point],
    report: VisumSQLiteReport,
    mode_mapping: Mapping[str, str],
    ignored_transport_systems: set[str],
    connector_epsilon_minutes: float,
    connector_capacity: float,
) -> gpd.GeoDataFrame:
    rows = []
    for (zone_no, node_no), group in connectors.groupby(["ZONENO", "NODENO"], sort=True):
        zone_no = int(zone_no)
        node_no = int(node_no)
        if zone_no not in point_by_zone or node_no not in point_by_node:
            report.add(
                "error",
                "missing-node-reference",
                "SQLite connector references missing zone or node coordinates",
                layer="connector",
                source_id=f"connector:{zone_no}:{node_no}",
            )
            continue
        by_direction = {str(row["DIRECTION"]): row for _, row in group.iterrows()}
        origin = by_direction.get("O")
        destination = by_direction.get("D")
        row = {
            "ZONENO": zone_no,
            "NODENO": node_no,
            "TSYSSET": _string_or_empty(origin.get("TSYSSET")) if origin is not None else "",
            "R_TSYSSET": _string_or_empty(destination.get("TSYSSET")) if destination is not None else "",
            "TYPENO": (
                origin.get("TYPENO")
                if origin is not None
                else (destination.get("TYPENO") if destination is not None else None)
            ),
            "LENGTH": _sqlite_length(origin.get("LENGTH")) if origin is not None else None,
            "R_LENGTH": _sqlite_length(destination.get("LENGTH")) if destination is not None else None,
            "V0PRT": None,
            "R_V0PRT": None,
            "CAPPRT": _sqlite_capacity(connector_capacity),
            "R_CAPPRT": _sqlite_capacity(connector_capacity),
            "T0PRT": _connector_time(
                origin,
                mode_mapping,
                ignored_transport_systems,
                report,
                zone_no,
                node_no,
                "O",
                connector_epsilon_minutes,
            ),
            "R_T0PRT": _connector_time(
                destination,
                mode_mapping,
                ignored_transport_systems,
                report,
                zone_no,
                node_no,
                "D",
                connector_epsilon_minutes,
            ),
            "geometry": LineString(
                [
                    (point_by_zone[zone_no].x, point_by_zone[zone_no].y),
                    (point_by_node[node_no].x, point_by_node[node_no].y),
                ]
            ),
        }
        rows.append(row)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _connector_time(
    row: pd.Series | None,
    mode_mapping: Mapping[str, str],
    ignored_transport_systems: set[str],
    report: VisumSQLiteReport,
    zone_no: int,
    node_no: int,
    direction: str,
    connector_epsilon_minutes: float,
) -> str | None:
    if row is None:
        return None
    values = []
    for token in _split_tsysset(row.get("TSYSSET")):
        token = token.upper()
        if token not in mode_mapping or token in ignored_transport_systems:
            continue
        field_name = f"T0_TSYS({token})"
        if field_name not in row or pd.isna(row[field_name]):
            continue
        seconds = float(row[field_name])
        if seconds == 0:
            report.add(
                "warning",
                "sqlite-zero-connector-time",
                "SQLite connector has explicit zero travel time; importing positive epsilon cost",
                layer="connector",
                field=field_name,
                source_id=f"connector:{zone_no}:{node_no}:{direction}",
            )
            values.append(connector_epsilon_minutes)
        elif seconds > 0:
            values.append(seconds / 60.0)
    if not values:
        return None
    return f"{max(values):.12f}".rstrip("0").rstrip(".") + "min"


def _count_location_layer(count_locations: pd.DataFrame, link_layer: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    link_points = {}
    for _, row in link_layer.iterrows():
        line = row.geometry
        link_points[int(row["NO"])] = line.interpolate(0.5, normalized=True)
    rows = []
    for _, row in count_locations.iterrows():
        source_link = row.get("LINKNO")
        geometry = (
            link_points.get(int(source_link))
            if pd.notna(source_link) and int(source_link) in link_points
            else Point(0, 0)
        )
        rows.append({**row.to_dict(), "geometry": geometry})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _read_zone_polygons(
    conn: sqlite3.Connection,
    zones: pd.DataFrame,
    transformer: Transformer,
    report: VisumSQLiteReport,
) -> dict[int, dict]:
    required = {"SURFACEITEM", "FACEITEM", "EDGE", "POINT"}
    available = {
        row[0].upper()
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')").fetchall()
    }
    if not required.issubset(available):
        return {}
    points = {
        int(row["ID"]): _transform_point(transformer, row["XCOORD"], row["YCOORD"])
        for row in conn.execute('SELECT ID, XCOORD, YCOORD FROM "POINT"')
    }
    edges = {
        int(row["ID"]): (int(row["FROMPOINTID"]), int(row["TOPOINTID"]))
        for row in conn.execute('SELECT ID, FROMPOINTID, TOPOINTID FROM "EDGE"')
    }
    edge_items: dict[int, list[Point]] = {}
    if "EDGEITEM" in available:
        for row in conn.execute('SELECT EDGEID, "INDEX", XCOORD, YCOORD FROM "EDGEITEM" ORDER BY EDGEID, "INDEX"'):
            edge_items.setdefault(int(row["EDGEID"]), []).append(
                _transform_point(transformer, row["XCOORD"], row["YCOORD"])
            )
    face_items: dict[int, list[tuple[int, int]]] = {}
    for row in conn.execute('SELECT FACEID, "INDEX", EDGEID, DIRECTION FROM "FACEITEM" ORDER BY FACEID, "INDEX"'):
        face_items.setdefault(int(row["FACEID"]), []).append((int(row["EDGEID"]), int(row["DIRECTION"] or 0)))
    surface_faces: dict[int, list[tuple[int, int]]] = {}
    for row in conn.execute('SELECT SURFACEID, FACEID, ENCLAVE FROM "SURFACEITEM"'):
        surface_faces.setdefault(int(row["SURFACEID"]), []).append((int(row["FACEID"]), int(row["ENCLAVE"] or 0)))

    polygons = {}
    for _, zone in zones.iterrows():
        zone_id = int(zone["NO"])
        surface_id = zone.get("SURFACEID")
        if pd.isna(surface_id) or int(surface_id) not in surface_faces:
            continue
        shells = []
        holes = []
        for face_id, enclave in surface_faces[int(surface_id)]:
            coords = _face_coordinates(face_id, face_items, edges, edge_items, points)
            if len(coords) < 4:
                report.add(
                    "warning",
                    "invalid-zone-surface",
                    "Zone surface could not be reconstructed",
                    layer="zone",
                    source_id=zone_id,
                )
                continue
            if enclave:
                holes.append(coords)
            else:
                shells.append(coords)
        if not shells:
            continue
        zone_polygons = [Polygon(shell, holes if idx == 0 else None) for idx, shell in enumerate(shells)]
        geometry = MultiPolygon(zone_polygons)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        polygons[zone_id] = {"NO": zone_id, "NAME": _string_or_empty(zone.get("NAME")), "geometry": geometry}
    return polygons


def _face_coordinates(
    face_id: int,
    face_items: Mapping[int, list[tuple[int, int]]],
    edges: Mapping[int, tuple[int, int]],
    edge_items: Mapping[int, list[Point]],
    points: Mapping[int, Point],
) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for edge_id, direction in face_items.get(face_id, []):
        if edge_id not in edges:
            continue
        from_point, to_point = edges[edge_id]
        edge_points = [points[from_point], *edge_items.get(edge_id, []), points[to_point]]
        if direction:
            edge_points = list(reversed(edge_points))
        edge_coords = [(point.x, point.y) for point in edge_points]
        if coords and edge_coords and coords[-1] == edge_coords[0]:
            coords.extend(edge_coords[1:])
        else:
            coords.extend(edge_coords)
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def _mapped_modes_from_value(value, mode_mapping: Mapping[str, str], ignored_transport_systems: set[str]) -> set[str]:
    modes = set()
    for token in _split_tsysset(value):
        token = token.upper()
        if token in ignored_transport_systems:
            continue
        mapped = mode_mapping.get(token)
        if mapped is not None:
            modes.add(mapped)
    return modes


def _string_or_empty(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _sqlite_length(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return f"{float(value):.12g}km"


def _sqlite_speed(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return f"{float(value):.12g}km/h"


def _sqlite_capacity(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return f"{float(value):.12g}veh/h"


def _inventory_sqlite_layers(layers: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, dict[str, object]]]:
    inventory = {}
    for layer, df in layers.items():
        fields = {}
        for column in df.columns:
            series = df[column]
            if column == "geometry":
                role = "geometry"
            elif column.startswith("R_"):
                role = "directional"
            elif column in {"NO", "FROMNODENO", "TONODENO", "ZONENO", "NODENO"}:
                role = "required"
            else:
                role = "optional"
            non_null = series.dropna()
            samples = [_jsonish(value) for value in non_null.head(3).tolist()]
            unique_values = []
            if column != "geometry" and non_null.nunique(dropna=True) <= 10:
                unique_values = [_jsonish(value) for value in non_null.unique().tolist()]
            fields[column] = {
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "unique_values": unique_values,
                "sample_values": samples,
                "unit_pattern": None,
                "role": role,
            }
        inventory[layer] = fields
    return inventory
