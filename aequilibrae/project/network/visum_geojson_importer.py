import math
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import LineString, MultiPolygon, Point

from aequilibrae.project.field_editor import FieldEditor
from aequilibrae.utils.db_utils import commit_and_close


REQUIRED_LAYERS = {"node", "link", "zone_centroid", "connector"}
OPTIONAL_LAYERS = {"zone_polygon", "countlocation"}
DEFERRED_LAYERS = {
    "stop",
    "stoppoint",
    "stop_point",
    "lineroute",
    "line_route",
    "ptline",
    "pt_line",
    "od_matrix",
    "matrix",
}
CONVENTIONAL_NAMES = {
    "node": ("node.geojson", "nodes.geojson"),
    "link": ("link.geojson", "links.geojson"),
    "zone_centroid": ("zone_centroid.geojson", "zonecentroid.geojson", "zone-centroid.geojson"),
    "zone_polygon": ("zone_polygon.geojson", "zone.geojson", "zones.geojson"),
    "connector": ("connector.geojson", "connectors.geojson"),
    "countlocation": ("countlocation.geojson", "count_location.geojson", "count-locations.geojson"),
}
DEFAULT_MODE_MAPPING = {"CAR": "c", "HGV": "h"}
CONNECTOR_FALLBACK_SPEED_KMH = 30.0
CONNECTOR_FALLBACK_CAPACITY = 99999.0
GEOD = Geod(ellps="WGS84")
COUNT_CANDIDATE_FIELDS = {"CAR_ORIG", "HVG_ORIG", "MOTOR_ORIG", "DTVW"}
DEFERRED_COUNT_FIELDS = {
    "CARS_LEFT",
    "CARS_RIGHT",
    "CARS_STRAIGHT",
    "CARS_PROJ",
    "HVG_PROJ",
    "MOTOR_PROJ",
}
_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


@dataclass
class VisumGeoJSONDiagnostic:
    """One VISUM GeoJSON import diagnostic."""

    severity: str
    code: str
    message: str
    layer: str | None = None
    field: str | None = None
    source_id: object | None = None


@dataclass
class VisumGeoJSONReport:
    """Diagnostics and provenance returned by a VISUM GeoJSON import."""

    diagnostics: list[VisumGeoJSONDiagnostic] = field(default_factory=list)
    discovered_layers: dict[str, str] = field(default_factory=dict)
    deferred_layers: list[str] = field(default_factory=list)
    field_inventory: dict[str, dict[str, dict[str, object]]] = field(default_factory=dict)
    imported_counts: dict[str, int] = field(default_factory=dict)
    source_references: dict[str, object] = field(default_factory=dict)
    mode_mapping: dict[str, str] = field(default_factory=dict)
    link_type_mapping: dict[object, str] = field(default_factory=dict)
    crs: dict[str, str] = field(default_factory=dict)

    @property
    def errors(self) -> list[VisumGeoJSONDiagnostic]:
        return [diag for diag in self.diagnostics if diag.severity == "error"]

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        layer: str | None = None,
        field: str | None = None,
        source_id: object | None = None,
    ) -> None:
        self.diagnostics.append(VisumGeoJSONDiagnostic(severity, code, message, layer, field, source_id))

    def raise_for_errors(self) -> None:
        if self.errors:
            messages = "; ".join(f"{diag.code}: {diag.message}" for diag in self.errors[:5])
            raise ValueError(f"VISUM GeoJSON import failed validation: {messages}")


def discover_visum_geojson_layers(path_or_layers: str | Path | Mapping[str, str | Path]) -> VisumGeoJSONReport:
    """Discover conventional VISUM GeoJSON layers or normalize an explicit layer-path mapping."""

    report = VisumGeoJSONReport()
    if isinstance(path_or_layers, Mapping):
        for layer, path in path_or_layers.items():
            key = _normalize_layer_name(layer)
            path = Path(path)
            if key in DEFERRED_LAYERS:
                report.deferred_layers.append(key)
            elif key in REQUIRED_LAYERS | OPTIONAL_LAYERS:
                report.discovered_layers[key] = str(path)
            else:
                report.add("warning", "unknown-layer", f"Layer '{layer}' is not recognized", layer=layer)
            if not path.exists():
                report.add("error", "missing-file", f"Layer file does not exist: {path}", layer=key)
    else:
        folder = Path(path_or_layers)
        if not folder.exists():
            report.add("error", "missing-input", f"VISUM GeoJSON path does not exist: {folder}")
            return report
        if not folder.is_dir():
            report.add("error", "invalid-input", "VISUM GeoJSON input must be a folder or explicit layer mapping")
            return report

        files = {file.name.lower(): file for file in folder.iterdir() if file.is_file()}
        for layer, names in CONVENTIONAL_NAMES.items():
            for name in names:
                if name in files:
                    report.discovered_layers[layer] = str(files[name])
                    break

        for file in folder.iterdir():
            stem = _normalize_layer_name(file.stem)
            if stem in DEFERRED_LAYERS or (file.suffix.lower() in {".omx", ".aem", ".mtx"} and "matrix" in stem):
                report.deferred_layers.append(stem)

    for layer in sorted(REQUIRED_LAYERS - set(report.discovered_layers)):
        report.add("error", "missing-layer", f"Required VISUM layer '{layer}' was not provided", layer=layer)

    for layer in sorted(set(report.deferred_layers)):
        report.add("warning", "deferred-layer", f"VISUM layer '{layer}' is recognized but deferred", layer=layer)

    report.deferred_layers = sorted(set(report.deferred_layers))
    return report


def read_visum_geojson_layers(
    layer_paths: Mapping[str, str | Path],
    report: VisumGeoJSONReport | None = None,
    *,
    source_crs: str | int | None = None,
    accept_default_crs: bool = False,
    default_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:4326",
) -> tuple[dict[str, gpd.GeoDataFrame], VisumGeoJSONReport]:
    """Read VISUM GeoJSON layers with explicit CRS handling."""

    report = report or VisumGeoJSONReport()
    layers = {}
    for layer, path in layer_paths.items():
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            if source_crs is not None:
                gdf = gdf.set_crs(source_crs)
                report.add("info", "source-crs-supplied", f"Using supplied CRS for layer '{layer}'", layer=layer)
            elif accept_default_crs:
                gdf = gdf.set_crs(default_crs)
                report.add(
                    "warning",
                    "default-crs-assumed",
                    f"Layer '{layer}' has no CRS metadata; assuming {default_crs}",
                    layer=layer,
                )
            else:
                report.add(
                    "error",
                    "missing-crs",
                    f"Layer '{layer}' has no CRS metadata; supply source_crs or accept_default_crs=True",
                    layer=layer,
                )
                layers[layer] = gdf
                continue
        if str(gdf.crs).upper() != target_crs.upper():
            report.crs[layer] = f"{gdf.crs} -> {target_crs}"
            gdf = gdf.to_crs(target_crs)
        else:
            report.crs[layer] = str(gdf.crs)
        layers[layer] = gdf
    return layers, report


def inventory_visum_layers(layers: Mapping[str, gpd.GeoDataFrame]) -> dict[str, dict[str, dict[str, object]]]:
    """Build a compact field inventory for diagnostics and reviewer workflows."""

    inventory = {}
    for layer, gdf in layers.items():
        fields = {}
        for column in gdf.columns:
            if column == gdf.geometry.name:
                role = "geometry"
            elif column.startswith("R_"):
                role = "directional"
            elif layer == "connector" and column == "NO":
                role = "optional"
            elif column in {"NO", "FROMNODENO", "TONODENO", "ZONENO", "NODENO"}:
                role = "required"
            elif column in COUNT_CANDIDATE_FIELDS:
                role = "count-candidate"
            elif column in DEFERRED_COUNT_FIELDS:
                role = "deferred"
            else:
                role = "optional"

            series = gdf[column]
            non_null = series.dropna()
            samples = [_jsonish(value) for value in non_null.head(3).tolist()]
            unique_values = []
            if column != gdf.geometry.name and non_null.nunique(dropna=True) <= 10:
                unique_values = [_jsonish(value) for value in non_null.unique().tolist()]
            fields[column] = {
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "unique_values": unique_values,
                "sample_values": samples,
                "unit_pattern": _unit_pattern(non_null),
                "role": role,
            }
        inventory[layer] = fields
    return inventory


def parse_visum_length(value) -> float | None:
    """Parse VISUM length values into meters."""

    return _parse_unit(value, {"m": 1.0, "meter": 1.0, "meters": 1.0, "km": 1000.0}, "m")


def parse_visum_speed(value) -> float | None:
    """Parse VISUM speed values into km/h."""

    return _parse_unit(
        value,
        {"km/h": 1.0, "kph": 1.0, "kmh": 1.0, "m/s": 3.6, "mph": 1.609344},
        "km/h",
    )


def parse_visum_time(value) -> float | None:
    """Parse VISUM time values into minutes."""

    return _parse_unit(
        value,
        {"min": 1.0, "mins": 1.0, "minute": 1.0, "s": 1.0 / 60.0, "sec": 1.0 / 60.0, "h": 60.0},
        "min",
    )


def parse_visum_capacity(value) -> float | None:
    """Parse VISUM capacity-like values into vehicles per hour."""

    return _parse_unit(
        value,
        {"veh/h": 1.0, "veh/hr": 1.0, "vehph": 1.0, "pcu/h": 1.0, "pc/h": 1.0, "/h": 1.0},
        "veh/h",
    )


class VisumGeoJSONImporter:
    def __init__(
        self,
        net,
        path_or_layers: str | Path | Mapping[str, str | Path],
        *,
        mode_mapping: Mapping[str, str] | None = None,
        ignored_transport_systems: set[str] | list[str] | tuple[str, ...] | None = None,
        link_type_mapping: Mapping[object, str] | None = None,
        source_crs: str | int | None = None,
        accept_default_crs: bool = False,
        allow_non_empty: bool = False,
        geometry_tolerance: float = 1e-6,
        duplicate_node_policy: str = "offset",
        duplicate_node_offset_meters: float = 0.25,
    ) -> None:
        if duplicate_node_policy not in {"offset", "error"}:
            raise ValueError("duplicate_node_policy must be 'offset' or 'error'")
        if duplicate_node_policy == "offset" and duplicate_node_offset_meters <= 0:
            raise ValueError("duplicate_node_offset_meters must be positive when duplicate_node_policy='offset'")
        self.net = net
        self.path_or_layers = path_or_layers
        self.mode_mapping = {str(k).upper(): v for k, v in (mode_mapping or DEFAULT_MODE_MAPPING).items()}
        self.ignored_transport_systems = {str(token).upper() for token in (ignored_transport_systems or set())}
        self.link_type_mapping = dict(link_type_mapping or {})
        self.source_crs = source_crs
        self.accept_default_crs = accept_default_crs
        self.allow_non_empty = allow_non_empty
        self.geometry_tolerance = geometry_tolerance
        self.duplicate_node_policy = duplicate_node_policy
        self.duplicate_node_offset_meters = duplicate_node_offset_meters
        self.report = VisumGeoJSONReport(mode_mapping=dict(self.mode_mapping))
        self.layers: dict[str, gpd.GeoDataFrame] = {}
        self.node_ids: dict[object, int] = {}
        self.node_points: dict[object, Point] = {}
        self.node_metadata: dict[object, dict[str, object]] = {}
        self.zone_points: dict[object, Point] = {}
        self.zone_metadata: dict[object, dict[str, object]] = {}
        self.source_to_link_id: dict[object, int] = {}
        self.connector_source_keys: dict[object, str] = {}
        self.connector_source_nos: dict[object, int | None] = {}
        self.skipped_records: dict[str, set[object]] = {"link": set(), "connector": set()}

    def doWork(self) -> VisumGeoJSONReport:
        self._discover_and_read()
        self._validate_required_columns()
        self._prepare_connector_source_keys()
        self._validate_mode_values()
        self._validate_assignment_values()
        self._validate_assignment_readiness()
        self._validate_topology()
        self._prepare_node_geometries()
        self._prepare_node_ids()
        self._prepare_zone_geometries()
        self.report.raise_for_errors()

        if self.net.count_links() > 0 and not self.allow_non_empty:
            raise FileExistsError("You can only import a VISUM GeoJSON network into an empty model file")

        self._prepare_database_fields()
        self._ensure_modes()
        link_type_by_value = self._ensure_link_types()
        self.report.link_type_mapping = {str(k): v for k, v in link_type_by_value.items()}

        with commit_and_close(self.net.project.path_to_file, spatial=True) as conn:
            conn.manual_transaction()
            with conn:
                node_ids = self._insert_nodes(conn)
                zone_ids = self._insert_zones_and_centroids(conn)
                link_ids = self._insert_links(conn, link_type_by_value)
                connector_ids = self._insert_connectors(conn)

        self.report.imported_counts.update(
            {
                "nodes": len(node_ids),
                "zones": len(zone_ids),
                "links": len(link_ids),
                "connectors": len(connector_ids),
            }
        )
        self._process_count_locations()
        self.report.add("info", "import-complete", "VISUM GeoJSON import completed")
        return self.report

    def _discover_and_read(self) -> None:
        discovery = discover_visum_geojson_layers(self.path_or_layers)
        self.report.discovered_layers.update(discovery.discovered_layers)
        self.report.deferred_layers.extend(discovery.deferred_layers)
        self.report.diagnostics.extend(discovery.diagnostics)
        self.report.raise_for_errors()

        self.layers, self.report = read_visum_geojson_layers(
            self.report.discovered_layers,
            self.report,
            source_crs=self.source_crs,
            accept_default_crs=self.accept_default_crs,
        )
        self.report.field_inventory = inventory_visum_layers(self.layers)

    def _validate_required_columns(self) -> None:
        required = {
            "node": {"NO"},
            "link": {"NO", "FROMNODENO", "TONODENO", "TSYSSET"},
            "zone_centroid": {"NO"},
            "connector": {"ZONENO", "NODENO", "TSYSSET"},
        }
        for layer, fields in required.items():
            if layer not in self.layers:
                continue
            missing = fields - set(self.layers[layer].columns)
            for field_name in sorted(missing):
                self.report.add(
                    "error",
                    "missing-field",
                    f"Layer '{layer}' is missing required field '{field_name}'",
                    layer=layer,
                    field=field_name,
                )

    def _prepare_connector_source_keys(self) -> None:
        if "connector" not in self.layers:
            return

        base_counts = {}
        for row_index, row in self.layers["connector"].iterrows():
            source_no = _optional_int(row.get("NO"))
            ab_modes = self._mapped_modes(row.get("TSYSSET"), "connector", source_no)
            ba_modes = self._mapped_modes(row.get("R_TSYSSET"), "connector", source_no)
            direction_code = _connector_direction_code(_direction(ab_modes, ba_modes))
            base_key = (
                f"connector:{_connector_key_part(row.get('ZONENO'))}:"
                f"{_connector_key_part(row.get('NODENO'))}:{direction_code}"
            )
            base_counts[base_key] = base_counts.get(base_key, 0) + 1
            suffix = "" if base_counts[base_key] == 1 else f":{base_counts[base_key]}"
            self.connector_source_keys[row_index] = f"{base_key}{suffix}"
            self.connector_source_nos[row_index] = source_no

    def _source_id(self, layer: str, row_index, row):
        if layer == "connector":
            return self._connector_source_key(row_index)
        return row.get("NO")

    def _connector_source_key(self, row_index) -> str:
        return self.connector_source_keys.get(row_index, f"connector:{row_index}")

    def _connector_source_no(self, row_index) -> int | None:
        return self.connector_source_nos.get(row_index)

    def _validate_topology(self) -> None:
        if not REQUIRED_LAYERS.issubset(self.layers):
            return

        nodes = self.layers["node"].set_index("NO")
        zones = self.layers["zone_centroid"].set_index("NO")
        for row_index, row in self.layers["link"].iterrows():
            if self._skip_record("link", row_index):
                continue
            source_id = row["NO"]
            if row["FROMNODENO"] not in nodes.index or row["TONODENO"] not in nodes.index:
                self.report.add(
                    "error",
                    "missing-node-reference",
                    "Link references missing node",
                    "link",
                    source_id=source_id,
                )
                continue
            self._validate_line_endpoint(row.geometry, nodes.loc[row["FROMNODENO"]].geometry, True, "link", source_id)
            self._validate_line_endpoint(row.geometry, nodes.loc[row["TONODENO"]].geometry, False, "link", source_id)

        for row_index, row in self.layers["connector"].iterrows():
            if self._skip_record("connector", row_index):
                continue
            source_id = self._connector_source_key(row_index)
            if row["ZONENO"] not in zones.index:
                self.report.add(
                    "error",
                    "missing-zone-reference",
                    "Connector references missing zone centroid",
                    "connector",
                    source_id=source_id,
                )
                continue
            if row["NODENO"] not in nodes.index:
                self.report.add(
                    "error",
                    "missing-node-reference",
                    "Connector references missing network node",
                    "connector",
                    source_id=source_id,
                )
                continue
            self._validate_line_endpoint(row.geometry, zones.loc[row["ZONENO"]].geometry, True, "connector", source_id)
            self._validate_line_endpoint(row.geometry, nodes.loc[row["NODENO"]].geometry, False, "connector", source_id)

    def _validate_mode_values(self) -> None:
        unmapped = {}
        ignored = {}
        for layer in ("link", "connector"):
            if layer not in self.layers:
                continue
            for row_index, row in self.layers[layer].iterrows():
                source_id = self._source_id(layer, row_index, row)
                ab_tokens = _split_tsysset(row.get("TSYSSET"))
                ba_tokens = _split_tsysset(row.get("R_TSYSSET"))
                tokens = ab_tokens + ba_tokens
                mapped_tokens = [token for token in tokens if token.upper() in self.mode_mapping]
                ignored_tokens = [token for token in tokens if token.upper() in self.ignored_transport_systems]
                unmapped_tokens = [
                    token
                    for token in tokens
                    if token.upper() not in self.mode_mapping and token.upper() not in self.ignored_transport_systems
                ]

                if not tokens:
                    self.skipped_records[layer].add(row_index)
                    self.report.add(
                        "warning",
                        "empty-transport-systems",
                        "Record has no declared transport systems and will not be imported",
                        layer=layer,
                        field="TSYSSET",
                        source_id=source_id,
                    )
                    continue

                for token in set(unmapped_tokens):
                    self._add_transport_system_summary(unmapped, layer, token, source_id)
                for token in set(ignored_tokens):
                    self._add_transport_system_summary(ignored, layer, token, source_id)

                if mapped_tokens:
                    continue
                if unmapped_tokens:
                    continue

                self.skipped_records[layer].add(row_index)
                self.report.add(
                    "warning",
                    "ignored-record",
                    "Record has only explicitly ignored transport systems and will not be imported",
                    layer=layer,
                    field="TSYSSET",
                    source_id=source_id,
                )

        for (layer, token), summary in sorted(unmapped.items()):
            source_text = ", ".join(str(source_id) for source_id in summary["sources"])
            self.report.add(
                "error",
                "unmapped-transport-system",
                (
                    f"Transport system '{token}' appears in {summary['count']} {layer} records and requires an "
                    f"explicit mode_mapping or ignored_transport_systems decision. Sample source IDs: {source_text}"
                ),
                layer=layer,
                field="TSYSSET",
            )

        for (layer, token), summary in sorted(ignored.items()):
            self.report.add(
                "info",
                "ignored-transport-system",
                f"Transport system '{token}' is explicitly ignored in {summary['count']} {layer} records",
                layer=layer,
                field="TSYSSET",
            )

    def _add_transport_system_summary(self, summary: dict, layer: str, token: str, source_id) -> None:
        key = (layer, token.upper())
        if key not in summary:
            summary[key] = {"count": 0, "sources": []}
        summary[key]["count"] += 1
        if len(summary[key]["sources"]) < 5:
            summary[key]["sources"].append(source_id)

    def _skip_record(self, layer: str, row_index) -> bool:
        return row_index in self.skipped_records.get(layer, set())

    def _validate_assignment_values(self) -> None:
        for layer in ("link", "connector"):
            if layer not in self.layers:
                continue
            for row_index, row in self.layers[layer].iterrows():
                if self._skip_record(layer, row_index):
                    continue
                source_id = self._source_id(layer, row_index, row)
                for prefix in ("", "R_"):
                    if not self._mapped_modes(row.get(f"{prefix}TSYSSET"), layer, source_id):
                        continue
                    for field_name, parser in (
                        (f"{prefix}LENGTH", parse_visum_length),
                        (f"{prefix}V0PRT", parse_visum_speed),
                        (f"{prefix}CAPPRT", parse_visum_capacity),
                        (f"{prefix}T0PRT", parse_visum_time),
                    ):
                        if field_name not in row or pd.isna(row[field_name]) or row[field_name] == "":
                            continue
                        try:
                            parser(row[field_name])
                        except ValueError as exc:
                            self.report.add(
                                "error",
                                "invalid-unit",
                                str(exc),
                                layer=layer,
                                field=field_name,
                                source_id=source_id,
                            )

    def _validate_assignment_readiness(self) -> None:
        for layer in ("link", "connector"):
            if layer not in self.layers:
                continue
            for row_index, row in self.layers[layer].iterrows():
                if self._skip_record(layer, row_index):
                    continue
                source_id = self._source_id(layer, row_index, row)
                for prefix, direction_value in (("", row.get("TSYSSET")), ("R_", row.get("R_TSYSSET"))):
                    if not self._mapped_modes(direction_value, layer, source_id):
                        continue
                    try:
                        length, speed, capacity, time = self._assignment_values(row, prefix, layer, source_id)
                    except ValueError:
                        continue
                    if time is None and length is not None and speed is not None and speed > 0:
                        time = (length / 1000.0) / speed * 60.0
                    if capacity is None:
                        self.report.add(
                            "warning",
                            "non-assignment-ready",
                            "Capacity is missing for an available private-traffic direction",
                            layer=layer,
                            field=f"{prefix}CAPPRT",
                            source_id=source_id,
                        )
                    if time is None:
                        self.report.add(
                            "warning",
                            "non-assignment-ready",
                            "Free-flow time cannot be parsed or derived for an available private-traffic direction",
                            layer=layer,
                            field=f"{prefix}T0PRT",
                            source_id=source_id,
                        )

    def _validate_line_endpoint(self, geometry, point, first: bool, layer: str, source_id) -> None:
        if geometry is None or geometry.is_empty or geometry.geom_type != "LineString":
            self.report.add("error", "invalid-geometry", "Expected LineString geometry", layer, source_id=source_id)
            return
        endpoint = geometry.coords[0 if first else -1]
        if point.distance(type(point)(endpoint)) > self.geometry_tolerance:
            self.report.add(
                "error",
                "endpoint-mismatch",
                "Geometry endpoint does not match referenced topology point",
                layer,
                source_id=source_id,
            )

    def _prepare_database_fields(self) -> None:
        self._add_fields(
            "nodes",
            {
                "visum_node_no": ("VISUM source node identifier", "INTEGER"),
                "visum_zone_no": ("VISUM source zone identifier for centroid nodes", "INTEGER"),
                "visum_original_lon": ("VISUM source node longitude before importer coordinate disambiguation", "REAL"),
                "visum_original_lat": ("VISUM source node latitude before importer coordinate disambiguation", "REAL"),
                "visum_xcoord": ("VISUM source projected X coordinate when provided", "REAL"),
                "visum_ycoord": ("VISUM source projected Y coordinate when provided", "REAL"),
                "visum_duplicate_coord_group": ("VISUM duplicate coordinate group identifier", "TEXT"),
                "visum_coord_offset_m": ("Approximate coordinate offset applied during VISUM import in meters", "REAL"),
            },
        )
        self._add_fields(
            "links",
            {
                "visum_link_no": ("VISUM source link identifier", "INTEGER"),
                "visum_connector_no": ("VISUM source connector identifier", "INTEGER"),
                "visum_connector_key": ("Deterministic VISUM connector source key", "TEXT"),
                "visum_length_ab": ("VISUM source AB length in meters", "NUMERIC"),
                "visum_length_ba": ("VISUM source BA length in meters", "NUMERIC"),
            },
        )
        self._add_fields("zones", {"visum_zone_no": ("VISUM source zone identifier", "INTEGER")})

    def _add_fields(self, table_name: str, fields: Mapping[str, tuple[str, str]]) -> None:
        editor = FieldEditor(self.net.project, table_name)
        existing = set(editor.all_fields())
        for field_name, (description, data_type) in fields.items():
            if field_name not in existing:
                editor.add(field_name, description=description, data_type=data_type)

    def _ensure_modes(self) -> None:
        for source_mode, mode_id in sorted(self.mode_mapping.items()):
            if len(mode_id) != 1:
                self.report.add(
                    "error",
                    "invalid-mode-id",
                    f"Mode mapping for {source_mode} must be a single-character AequilibraE mode ID",
                    field=source_mode,
                )
                continue
            if mode_id not in self.net.modes.all_modes():
                mode = self.net.modes.new(mode_id)
                mode.mode_name = "hgv" if source_mode == "HGV" else source_mode.lower()
                mode.description = f"VISUM transport system {source_mode}"
                self.net.modes.add(mode)
                mode.save()
        self.report.raise_for_errors()

    def _ensure_link_types(self) -> dict[object, str]:
        source_values = []
        for layer in ("link", "connector"):
            if layer not in self.layers:
                continue
            if layer == "connector":
                continue
            for row_index, row in self.layers[layer].iterrows():
                if self._skip_record(layer, row_index):
                    continue
                source_values.append(_link_type_source_value(row))
                source_values.append(_link_type_source_value(row, "R_"))
        mapping = dict(self.link_type_mapping)
        used_ids = set(self.net.link_types.all_types())
        existing_by_name = {
            link_type.link_type.lower(): link_type.link_type for link_type in self.net.link_types.all_types().values()
        }

        for source in sorted(set(source_values), key=str):
            if source in mapping:
                name = _clean_name(mapping[source])
            else:
                name = _clean_name(f"visum_{source}")
                mapping[source] = name
            if name.lower() in existing_by_name:
                mapping[source] = existing_by_name[name.lower()]
                continue

            link_type_id = _first_unused_link_type_id(str(source), used_ids)
            if link_type_id is None:
                self.report.add(
                    "error",
                    "link-type-id-exhausted",
                    "No available link-type IDs remain",
                    field=str(source),
                )
                continue
            link_type = self.net.link_types.new(link_type_id)
            link_type.link_type = name
            link_type.description = f"VISUM link type/class {source}"
            link_type.save()
            used_ids.add(link_type_id)
            mapping[source] = name

        self.report.raise_for_errors()
        return mapping

    def _insert_nodes(self, conn) -> dict[object, int]:
        mapping = {}
        query = """
            INSERT INTO nodes(
                node_id, is_centroid, visum_node_no, visum_original_lon, visum_original_lat, visum_xcoord,
                visum_ycoord, visum_duplicate_coord_group, visum_coord_offset_m, geometry
            )
            VALUES(?, 0, ?, ?, ?, ?, ?, ?, ?, MakePoint(?, ?, 4326))
        """
        for _, row in self.layers["node"].iterrows():
            node_id = self._node_id(row["NO"])
            point = self._node_point(row["NO"])
            metadata = self.node_metadata.get(row["NO"], {})
            conn.execute(
                query,
                (
                    node_id,
                    int(row["NO"]),
                    metadata.get("original_lon", row.geometry.x),
                    metadata.get("original_lat", row.geometry.y),
                    metadata.get("xcoord"),
                    metadata.get("ycoord"),
                    metadata.get("duplicate_group"),
                    metadata.get("offset_m", 0.0),
                    point.x,
                    point.y,
                ),
            )
            mapping[row["NO"]] = node_id
        self.report.source_references["nodes"] = mapping
        return mapping

    def _insert_zones_and_centroids(self, conn) -> dict[object, int]:
        zones = {}
        polygons = {}
        if "zone_polygon" in self.layers:
            polygons = {row["NO"]: row for _, row in self.layers["zone_polygon"].iterrows()}
        for _, row in self.layers["zone_centroid"].iterrows():
            zone_id = int(row["NO"])
            name = str(row["NAME"]) if "NAME" in row and not pd.isna(row["NAME"]) else ""
            polygon_row = polygons.get(row["NO"])
            geometry = polygon_row.geometry if polygon_row is not None else MultiPolygon([row.geometry.buffer(1e-6)])
            if geometry.geom_type == "Polygon":
                geometry = MultiPolygon([geometry])
            conn.execute(
                """
                INSERT INTO zones(zone_id, name, visum_zone_no, geometry)
                VALUES(?, ?, ?, GeomFromText(?, 4326))
                """,
                (zone_id, name, zone_id, geometry.wkt),
            )
            point = self._zone_point(row["NO"])
            metadata = self.zone_metadata.get(row["NO"], {})
            conn.execute(
                """
                INSERT INTO nodes(
                    node_id, is_centroid, visum_zone_no, visum_original_lon, visum_original_lat, visum_xcoord,
                    visum_ycoord, visum_duplicate_coord_group, visum_coord_offset_m, geometry
                )
                VALUES(?, 1, ?, ?, ?, ?, ?, ?, ?, MakePoint(?, ?, 4326))
                """,
                (
                    zone_id,
                    zone_id,
                    metadata.get("original_lon", row.geometry.x),
                    metadata.get("original_lat", row.geometry.y),
                    metadata.get("xcoord"),
                    metadata.get("ycoord"),
                    metadata.get("duplicate_group"),
                    metadata.get("offset_m", 0.0),
                    point.x,
                    point.y,
                ),
            )
            zones[row["NO"]] = zone_id
        self.report.source_references["zones"] = zones
        return zones

    def _insert_links(self, conn, link_type_by_value: Mapping[object, str]) -> dict[object, int]:
        mapping = {}
        next_link_id = self._next_link_id(conn)
        query = """
            INSERT INTO links(
                link_id, a_node, b_node, direction, modes, link_type, speed_ab, speed_ba, capacity_ab, capacity_ba,
                travel_time_ab, travel_time_ba, visum_link_no, visum_length_ab, visum_length_ba, geometry
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GeomFromText(?, 4326))
        """
        for row_index, row in self.layers["link"].iterrows():
            if self._skip_record("link", row_index):
                continue
            source_id = int(row["NO"])
            ab_modes = self._mapped_modes(row.get("TSYSSET"), "link", source_id)
            ba_modes = self._mapped_modes(row.get("R_TSYSSET"), "link", source_id)
            length_ab, speed_ab, capacity_ab, time_ab = self._assignment_values(row, "", "link", source_id)
            length_ba, speed_ba, capacity_ba, time_ba = self._assignment_values(row, "R_", "link", source_id)
            if ab_modes and ba_modes and ab_modes != ba_modes:
                records = [
                    (1, ab_modes, speed_ab, None, capacity_ab, None, time_ab, None, length_ab, None, ""),
                    (-1, ba_modes, None, speed_ba, None, capacity_ba, None, time_ba, None, length_ba, "R_"),
                ]
                self.report.add(
                    "info",
                    "directional-mode-split",
                    "Link has different mode sets by direction and was imported as directional one-way records",
                    layer="link",
                    field="TSYSSET",
                    source_id=source_id,
                )
            else:
                direction = _direction(ab_modes, ba_modes)
                if direction == 1:
                    speed_ba = capacity_ba = time_ba = None
                elif direction == -1:
                    speed_ab = capacity_ab = time_ab = None
                records = [
                    (
                        direction,
                        ab_modes | ba_modes,
                        speed_ab,
                        speed_ba,
                        capacity_ab,
                        capacity_ba,
                        time_ab,
                        time_ba,
                        length_ab,
                        length_ba,
                        "",
                    )
                ]
            for (
                direction,
                modes,
                rec_speed_ab,
                rec_speed_ba,
                rec_capacity_ab,
                rec_capacity_ba,
                rec_time_ab,
                rec_time_ba,
                rec_length_ab,
                rec_length_ba,
                type_prefix,
            ) in records:
                link_id = next_link_id
                next_link_id += 1
                link_type = link_type_by_value[_link_type_source_value(row, type_prefix)]
                conn.execute(
                    query,
                    (
                        link_id,
                        self._node_id(row["FROMNODENO"]),
                        self._node_id(row["TONODENO"]),
                        direction,
                        "".join(sorted(modes)),
                        link_type,
                        rec_speed_ab,
                        rec_speed_ba,
                        rec_capacity_ab,
                        rec_capacity_ba,
                        rec_time_ab,
                        rec_time_ba,
                        source_id,
                        rec_length_ab,
                        rec_length_ba,
                        self._link_geometry(row).wkt,
                    ),
                )
                mapping.setdefault(row["NO"], link_id)
                self.source_to_link_id.setdefault(row["NO"], link_id)
        self.report.source_references["links"] = mapping
        return mapping

    def _insert_connectors(self, conn) -> dict[object, int]:
        mapping = {}
        next_link_id = self._next_link_id(conn)
        query = """
            INSERT INTO links(
                link_id, a_node, b_node, direction, modes, link_type, speed_ab, speed_ba, capacity_ab, capacity_ba,
                travel_time_ab, travel_time_ba, visum_connector_no, visum_connector_key, visum_length_ab,
                visum_length_ba, geometry
            )
            VALUES(?, ?, ?, ?, ?, 'centroid_connector', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GeomFromText(?, 4326))
        """
        for row_index, row in self.layers["connector"].iterrows():
            if self._skip_record("connector", row_index):
                continue
            source_id = self._connector_source_key(row_index)
            source_no = self._connector_source_no(row_index)
            ab_modes = self._mapped_modes(row.get("TSYSSET"), "connector", source_id)
            ba_modes = self._mapped_modes(row.get("R_TSYSSET"), "connector", source_id)
            length_ab, speed_ab, capacity_ab, time_ab = self._assignment_values(row, "", "connector", source_id)
            length_ba, speed_ba, capacity_ba, time_ba = self._assignment_values(row, "R_", "connector", source_id)
            if ab_modes and ba_modes and ab_modes != ba_modes:
                records = [
                    (1, ab_modes, speed_ab, None, capacity_ab, None, time_ab, None, length_ab, None),
                    (-1, ba_modes, None, speed_ba, None, capacity_ba, None, time_ba, None, length_ba),
                ]
                self.report.add(
                    "info",
                    "directional-mode-split",
                    "Connector has different mode sets by direction and was imported as directional one-way records",
                    layer="connector",
                    field="TSYSSET",
                    source_id=source_id,
                )
            else:
                direction = _direction(ab_modes, ba_modes)
                if direction == 1:
                    speed_ba = capacity_ba = time_ba = None
                elif direction == -1:
                    speed_ab = capacity_ab = time_ab = None
                records = [
                    (
                        direction,
                        ab_modes | ba_modes,
                        speed_ab,
                        speed_ba,
                        capacity_ab,
                        capacity_ba,
                        time_ab,
                        time_ba,
                        length_ab,
                        length_ba,
                    )
                ]
            for (
                direction,
                modes,
                rec_speed_ab,
                rec_speed_ba,
                rec_capacity_ab,
                rec_capacity_ba,
                rec_time_ab,
                rec_time_ba,
                rec_length_ab,
                rec_length_ba,
            ) in records:
                link_id = next_link_id
                next_link_id += 1
                conn.execute(
                    query,
                    (
                        link_id,
                        int(row["ZONENO"]),
                        self._node_id(row["NODENO"]),
                        direction,
                        "".join(sorted(modes)),
                        rec_speed_ab,
                        rec_speed_ba,
                        rec_capacity_ab,
                        rec_capacity_ba,
                        rec_time_ab,
                        rec_time_ba,
                        source_no,
                        source_id,
                        rec_length_ab,
                        rec_length_ba,
                        self._connector_geometry(row).wkt,
                    ),
                )
                mapping.setdefault(source_id, link_id)
        self.report.source_references["connectors"] = mapping
        return mapping

    def _next_link_id(self, conn) -> int:
        return int(conn.execute("SELECT COALESCE(MAX(link_id), 0) + 1 FROM links").fetchone()[0])

    def _prepare_node_geometries(self) -> None:
        if "node" not in self.layers:
            return

        groups: dict[tuple[float, float], list[tuple[object, object, Point]]] = {}
        for row_index, row in self.layers["node"].iterrows():
            point = row.geometry
            if point is None or point.is_empty:
                continue
            source_id = row["NO"]
            self.node_points[source_id] = point
            self.node_metadata[source_id] = {
                "original_lon": point.x,
                "original_lat": point.y,
                "xcoord": _optional_float(row.get("XCOORD")),
                "ycoord": _optional_float(row.get("YCOORD")),
                "duplicate_group": None,
                "offset_m": 0.0,
            }
            groups.setdefault((point.x, point.y), []).append((row_index, source_id, point))

        duplicate_group_number = 0
        for records in groups.values():
            if len(records) < 2:
                continue
            duplicate_group_number += 1
            source_ids = [record[1] for record in records]
            if self.duplicate_node_policy == "error":
                self.report.add(
                    "error",
                    "coincident-node-coordinate",
                    "VISUM nodes share identical coordinates; use duplicate_node_policy='offset' to preserve topology",
                    layer="node",
                    source_id=", ".join(str(source_id) for source_id in source_ids),
                )
                continue

            group_id = f"node-coordinate-{duplicate_group_number}"
            for position, (_, source_id, point) in enumerate(records):
                offset_m = 0.0
                adjusted_point = point
                if position > 0:
                    offset_m = self.duplicate_node_offset_meters
                    adjusted_point = _offset_point(point, position - 1, len(records) - 1, offset_m)
                self.node_points[source_id] = adjusted_point
                self.node_metadata[source_id]["duplicate_group"] = group_id
                self.node_metadata[source_id]["offset_m"] = offset_m

            self.report.add(
                "warning",
                "coincident-node-offset",
                (
                    f"{len(records)} VISUM nodes share coordinates; importer applied deterministic offsets to "
                    "preserve source topology. Source node IDs: "
                    f"{', '.join(str(source_id) for source_id in source_ids)}"
                ),
                layer="node",
                source_id=", ".join(str(source_id) for source_id in source_ids),
            )

        self.report.source_references["node_coordinate_offsets"] = {
            _jsonish(source_id): {
                "original_lon": metadata["original_lon"],
                "original_lat": metadata["original_lat"],
                "offset_m": metadata["offset_m"],
                "duplicate_group": metadata["duplicate_group"],
            }
            for source_id, metadata in self.node_metadata.items()
            if metadata.get("duplicate_group") is not None
        }

    def _node_point(self, source_id) -> Point:
        return self.node_points[source_id]

    def _prepare_node_ids(self) -> None:
        if "node" not in self.layers:
            return

        zone_ids = {int(row["NO"]) for _, row in self.layers.get("zone_centroid", pd.DataFrame()).iterrows()}
        source_node_ids = [int(row["NO"]) for _, row in self.layers["node"].iterrows()]
        used_ids = set(zone_ids)
        next_node_id = max(used_ids | set(source_node_ids), default=0) + 1

        for _, row in self.layers["node"].iterrows():
            source_id = row["NO"]
            preferred_id = int(source_id)
            if preferred_id not in used_ids:
                node_id = preferred_id
            else:
                while next_node_id in used_ids:
                    next_node_id += 1
                node_id = next_node_id
                next_node_id += 1
                self.report.add(
                    "warning",
                    "node-id-remapped",
                    (
                        f"VISUM node {source_id} uses an ID reserved for a zone centroid; "
                        f"imported as AequilibraE node {node_id}"
                    ),
                    layer="node",
                    field="NO",
                    source_id=source_id,
                )
            self.node_ids[source_id] = node_id
            used_ids.add(node_id)

    def _node_id(self, source_id) -> int:
        return self.node_ids[source_id]

    def _prepare_zone_geometries(self) -> None:
        if "zone_centroid" not in self.layers:
            return

        occupied = {(point.x, point.y) for point in self.node_points.values()}
        offset_number = 0
        for _, row in self.layers["zone_centroid"].iterrows():
            source_id = row["NO"]
            point = row.geometry
            self.zone_metadata[source_id] = {
                "original_lon": point.x,
                "original_lat": point.y,
                "xcoord": _optional_float(row.get("XCOORD")),
                "ycoord": _optional_float(row.get("YCOORD")),
                "duplicate_group": None,
                "offset_m": 0.0,
            }

            adjusted_point = point
            if (adjusted_point.x, adjusted_point.y) in occupied:
                offset_number += 1
                group_id = f"zone-coordinate-{offset_number}"
                offset_step = 0
                while (adjusted_point.x, adjusted_point.y) in occupied:
                    adjusted_point = _offset_point(point, offset_step, 8, self.duplicate_node_offset_meters)
                    offset_step += 1
                self.zone_metadata[source_id]["duplicate_group"] = group_id
                self.zone_metadata[source_id]["offset_m"] = self.duplicate_node_offset_meters
                self.report.add(
                    "warning",
                    "coincident-centroid-offset",
                    (
                        f"VISUM zone centroid {source_id} shares coordinates with another imported node; "
                        "importer applied a deterministic offset"
                    ),
                    layer="zone_centroid",
                    source_id=source_id,
                )

            self.zone_points[source_id] = adjusted_point
            occupied.add((adjusted_point.x, adjusted_point.y))

        self.report.source_references["zone_coordinate_offsets"] = {
            _jsonish(source_id): {
                "original_lon": metadata["original_lon"],
                "original_lat": metadata["original_lat"],
                "offset_m": metadata["offset_m"],
                "duplicate_group": metadata["duplicate_group"],
            }
            for source_id, metadata in self.zone_metadata.items()
            if metadata.get("duplicate_group") is not None
        }

    def _zone_point(self, source_id) -> Point:
        return self.zone_points[source_id]

    def _link_geometry(self, row) -> LineString:
        return _line_with_endpoints(
            row.geometry,
            self._node_point(row["FROMNODENO"]),
            self._node_point(row["TONODENO"]),
        )

    def _connector_geometry(self, row) -> LineString:
        return _line_with_endpoints(row.geometry, self._zone_point(row["ZONENO"]), self._node_point(row["NODENO"]))

    def _mapped_modes(self, value, layer: str, source_id) -> set[str]:
        modes = set()
        for token in _split_tsysset(value):
            mapped = self.mode_mapping.get(token.upper())
            if mapped is not None:
                modes.add(mapped)
        return modes

    def _assignment_values(
        self, row, prefix: str, layer: str, source_id
    ) -> tuple[float | None, float | None, float | None, float | None]:
        length = parse_visum_length(row.get(f"{prefix}LENGTH"))
        speed = parse_visum_speed(row.get(f"{prefix}V0PRT"))
        capacity = parse_visum_capacity(row.get(f"{prefix}CAPPRT"))
        time = parse_visum_time(row.get(f"{prefix}T0PRT"))
        if layer == "connector":
            if length is None or length <= 0:
                geometry_length = _geodesic_length(row.geometry)
                if geometry_length is not None and geometry_length > 0:
                    length = geometry_length
                    self.report.add(
                        "warning",
                        "connector-length-defaulted",
                        "Connector length defaulted from geometry length",
                        layer=layer,
                        field=f"{prefix}LENGTH",
                        source_id=source_id,
                    )
            if time is not None and time <= 0:
                time = None
            if speed is not None and speed <= 0:
                speed = None
            if time is None and speed is None and length is not None and length > 0:
                speed = CONNECTOR_FALLBACK_SPEED_KMH
                self.report.add(
                    "warning",
                    "connector-speed-defaulted",
                    f"Connector speed defaulted to {CONNECTOR_FALLBACK_SPEED_KMH:g} km/h",
                    layer=layer,
                    field=f"{prefix}V0PRT",
                    source_id=source_id,
                )
            if capacity is None or capacity <= 0:
                capacity = CONNECTOR_FALLBACK_CAPACITY
                self.report.add(
                    "warning",
                    "connector-capacity-defaulted",
                    f"Connector capacity defaulted to {CONNECTOR_FALLBACK_CAPACITY:g}",
                    layer=layer,
                    field=f"{prefix}CAPPRT",
                    source_id=source_id,
                )
        if time is None and length is not None and speed is not None and speed > 0:
            time = (length / 1000.0) / speed * 60.0
        for field_name, parsed in ((f"{prefix}V0PRT", speed), (f"{prefix}CAPPRT", capacity), (f"{prefix}T0PRT", time)):
            if parsed is not None and parsed <= 0:
                self.report.add(
                    "warning",
                    "non-assignment-ready",
                    f"Field '{field_name}' is not positive for assignment",
                    layer=layer,
                    field=field_name,
                    source_id=source_id,
                )
        return length, speed, capacity, time

    def _process_count_locations(self) -> None:
        associations = []
        if "countlocation" not in self.layers:
            self.report.source_references["count_locations"] = associations
            return
        for _, row in self.layers["countlocation"].iterrows():
            source_link = row.get("LINKNO")
            if source_link in self.source_to_link_id:
                count_fields = {
                    field_name: _jsonish(row[field_name])
                    for field_name in COUNT_CANDIDATE_FIELDS
                    if field_name in row and not pd.isna(row[field_name])
                }
                associations.append(
                    {
                        "source_id": _jsonish(row.get("NO")),
                        "link_id": self.source_to_link_id[source_link],
                        "counts": count_fields,
                    }
                )
            else:
                self.report.add(
                    "warning",
                    "unresolved-count-link",
                    "Count location could not be associated with an imported link",
                    layer="countlocation",
                    field="LINKNO",
                    source_id=_jsonish(row.get("NO")),
                )
            deferred = sorted(
                field_name for field_name in DEFERRED_COUNT_FIELDS if field_name in row and not pd.isna(row[field_name])
            )
            if deferred:
                self.report.add(
                    "warning",
                    "deferred-count-fields",
                    f"Count fields deferred for future traffic-data workflows: {', '.join(deferred)}",
                    layer="countlocation",
                    source_id=_jsonish(row.get("NO")),
                )
        self.report.source_references["count_locations"] = associations


def _parse_unit(value, multipliers: Mapping[str, float], default_unit: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, int | float):
        return float(value) * multipliers[default_unit]
    text = str(value).strip().lower().replace(" ", "")
    if text in {"", "nan", "none", "null"}:
        return None
    match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)([a-z/]+)?", text)
    if match is None:
        raise ValueError(f"Could not parse VISUM unit value '{value}'")
    number = float(match.group(1))
    unit = match.group(2) or default_unit
    if unit not in multipliers:
        raise ValueError(f"Unsupported VISUM unit '{unit}' in value '{value}'")
    return number * multipliers[unit]


def _split_tsysset(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [token for token in re.split(r"[,;|\s]+", str(value).strip()) if token]


def _direction(ab_modes: set[str], ba_modes: set[str]) -> int:
    if ab_modes and ba_modes:
        return 0
    if ab_modes:
        return 1
    if ba_modes:
        return -1
    return 1


def _connector_direction_code(direction: int) -> str:
    return {0: "B", 1: "O", -1: "D"}[direction]


def _connector_key_part(value) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _optional_int(value) -> int | None:
    if value is None or pd.isna(value) or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value) -> float | None:
    if value is None or pd.isna(value) or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _offset_point(point: Point, offset_index: int, offset_count: int, offset_meters: float) -> Point:
    angle = 2.0 * math.pi * offset_index / max(offset_count, 1)
    latitude_radians = math.radians(point.y)
    meters_per_degree_latitude = 111_320.0
    meters_per_degree_longitude = max(math.cos(latitude_radians) * meters_per_degree_latitude, 1e-9)
    dx = math.cos(angle) * offset_meters / meters_per_degree_longitude
    dy = math.sin(angle) * offset_meters / meters_per_degree_latitude
    return Point(point.x + dx, point.y + dy)


def _line_with_endpoints(geometry, start_point: Point, end_point: Point) -> LineString:
    coords = [(coord[0], coord[1]) for coord in geometry.coords]
    coords[0] = (start_point.x, start_point.y)
    coords[-1] = (end_point.x, end_point.y)
    return LineString(coords)


def _geodesic_length(geometry) -> float | None:
    if geometry is None or geometry.is_empty or geometry.geom_type != "LineString":
        return None
    coords = list(geometry.coords)
    if len(coords) < 2:
        return None
    length = 0.0
    for start, end in zip(coords[:-1], coords[1:], strict=True):
        _, _, distance = GEOD.inv(start[0], start[1], end[0], end[1])
        length += distance
    return length


def _link_type_source_value(row, prefix: str = "") -> object:
    for field_name in (f"{prefix}LC", f"{prefix}TYPENO"):
        if field_name in row and not pd.isna(row[field_name]) and row[field_name] != "":
            return row[field_name]
    return "default"


def _first_unused_link_type_id(source: str, used: set[str]) -> str | None:
    candidates = _clean_name(source) + string.ascii_lowercase + string.ascii_uppercase
    for candidate in candidates:
        if candidate in string.ascii_letters and candidate not in used:
            return candidate
    return None


def _clean_name(value) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\d", lambda match: f"_{_DIGIT_WORDS[match.group(0)]}", text)
    text = re.sub(r"[^a-zA-Z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text or text[0] not in string.ascii_letters:
        text = f"visum_{text}" if text else "visum_default"
    return text


def _normalize_layer_name(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _unit_pattern(series: pd.Series) -> str | None:
    for value in series.head(10):
        if isinstance(value, str):
            match = re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*([a-zA-Z/]+)\s*", value)
            if match:
                return match.group(1)
    return None


def _jsonish(value):
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value
