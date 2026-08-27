"""Pure-Python replacement for the mod_spatialite SQLite extension.

AequilibraE historically loaded the native SpatiaLite extension on every database
connection, which required a Debian/Homebrew package on Linux/macOS and a DLL
download on Windows. This module reimplements the subset of SpatiaLite that
AequilibraE actually uses as application-defined SQL functions (UDFs) backed by
shapely and pyproj, registered per-connection with ``sqlite3``. The spatial index
uses SQLite's built-in R*Tree module, which is exactly what SpatiaLite itself
uses under the hood — so databases written this way remain byte-compatible,
fully valid SpatiaLite databases that QGIS and other tools can open unchanged.

Two SpatiaLite features cannot be replicated from Python and are handled
differently:

* The ``SpatialIndex`` virtual table (an MBR-query wrapper over the ``idx_*``
  R*Tree tables). All queries are rewritten to hit the R*Tree tables directly;
  triggers in pre-existing databases that reference ``SpatialIndex`` are
  transparently rewritten on first (writable) connection.
* ``InitSpatialMetaData()`` — new projects are created from a pre-initialised
  template database, so only a minimal implementation is provided.
"""

import re
import sqlite3

import shapely
import shapely.wkt

from aequilibrae.utils.gaia_geometry import (
    gaia_geometry_code,
    gaia_mbr,
    gaia_point_xy,
    gaia_srid,
    gaia_to_shapely,
    gaia_to_wkb,
    linestring_boundary_point,
    linestring_lonlats,
    make_point_blob,
    shapely_to_gaia,
)

GEOMETRY_TYPE_CODES = {
    "GEOMETRY": 0,
    "POINT": 1,
    "LINESTRING": 2,
    "POLYGON": 3,
    "MULTIPOINT": 4,
    "MULTILINESTRING": 5,
    "MULTIPOLYGON": 6,
    "GEOMETRYCOLLECTION": 7,
}

_MULTI_CAST = {
    "Point": shapely.MultiPoint,
    "LineString": shapely.MultiLineString,
    "Polygon": shapely.MultiPolygon,
}

_geod = None


def _wgs84_geod():
    global _geod
    if _geod is None:
        from pyproj import Geod

        _geod = Geod(ellps="WGS84")
    return _geod


def _dims_code(dims) -> int:
    if isinstance(dims, int):
        return 1000 if dims == 3 else 0
    dims = str(dims).upper()
    return {"XY": 0, "XYZ": 1000, "XYM": 2000, "XYZM": 3000}.get(dims, 0)


# ---------------------------------------------------------------------------
# Scalar geometry functions (pure, no connection state)
# ---------------------------------------------------------------------------


def _geom_from_wkb(wkb, srid=0):
    if wkb is None:
        return None
    try:
        return shapely_to_gaia(shapely.from_wkb(bytes(wkb)), int(srid))
    except Exception:
        return None


def _geom_from_text(wkt, srid=0):
    if wkt is None:
        return None
    try:
        return shapely_to_gaia(shapely.wkt.loads(wkt), int(srid))
    except Exception:
        return None


def _make_point(x, y, srid=0):
    if x is None or y is None:
        return None
    return make_point_blob(float(x), float(y), int(srid))


def _make_line(a, b):
    if a is None or b is None:
        return None
    try:
        ga, gb = gaia_to_shapely(bytes(a)), gaia_to_shapely(bytes(b))
        line = shapely.LineString(list(ga.coords) + list(gb.coords))
        return shapely_to_gaia(line, gaia_srid(bytes(a)))
    except Exception:
        return None


def _st_multi(blob):
    if blob is None:
        return None
    try:
        blob = bytes(blob)
        geom = gaia_to_shapely(blob)
        caster = _MULTI_CAST.get(geom.geom_type)
        if caster is not None:
            geom = caster([geom])
        return shapely_to_gaia(geom, gaia_srid(blob))
    except Exception:
        return None


def _as_binary(blob):
    if blob is None:
        return None
    try:
        return gaia_to_wkb(bytes(blob))
    except Exception:
        return None


_TYPE_PAREN_RE = re.compile(r"\b([A-Z]+) \(")
_MULTIPOINT_RE = re.compile(r"MULTIPOINT\((.*)\)$")


def _as_text(blob):
    if blob is None:
        return None
    try:
        wkt = shapely.to_wkt(gaia_to_shapely(bytes(blob)), rounding_precision=-1)
        # SpatiaLite writes no space between type names and coordinates,
        # and no parentheses around individual MULTIPOINT members
        wkt = _TYPE_PAREN_RE.sub(r"\1(", wkt)
        if wkt.startswith("MULTIPOINT"):
            wkt = _MULTIPOINT_RE.sub(lambda m: "MULTIPOINT(" + m.group(1).replace("(", "").replace(")", "") + ")", wkt)
        return wkt
    except Exception:
        return None


def _st_x(blob):
    if blob is None:
        return None
    try:
        return gaia_point_xy(bytes(blob))[0]
    except Exception:
        return None


def _st_y(blob):
    if blob is None:
        return None
    try:
        return gaia_point_xy(bytes(blob))[1]
    except Exception:
        return None


def _srid(blob):
    if blob is None:
        return None
    try:
        return gaia_srid(bytes(blob))
    except Exception:
        return None


def _start_point(blob):
    return None if blob is None else linestring_boundary_point(bytes(blob), start=True)


def _end_point(blob):
    return None if blob is None else linestring_boundary_point(bytes(blob), start=False)


def _set_boundary_point(line_blob, point_blob, start: bool):
    if line_blob is None or point_blob is None:
        return None
    try:
        line_blob = bytes(line_blob)
        line = gaia_to_shapely(line_blob)
        if line.geom_type != "LineString":
            return None
        coords = list(line.coords)
        new_pt = gaia_point_xy(bytes(point_blob))
        if start:
            coords[0] = new_pt
        else:
            coords[-1] = new_pt
        return shapely_to_gaia(shapely.LineString(coords), gaia_srid(line_blob))
    except Exception:
        return None


def _set_start_point(line_blob, point_blob):
    return _set_boundary_point(line_blob, point_blob, start=True)


def _set_end_point(line_blob, point_blob):
    return _set_boundary_point(line_blob, point_blob, start=False)


def _geodesic_length(blob):
    if blob is None:
        return None
    try:
        geod = _wgs84_geod()

        fast = linestring_lonlats(bytes(blob))
        if fast is not None:  # linestrings are the overwhelmingly common case in triggers
            return geod.line_length(*fast)

        def measure(geom):
            if geom.geom_type == "Polygon":
                # native SpatiaLite measures every ring, including interior ones
                return geod.geometry_length(geom.exterior) + sum(
                    geod.geometry_length(ring) for ring in geom.interiors
                )
            if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
                return sum(measure(g) for g in geom.geoms)
            return geod.geometry_length(geom)

        return measure(gaia_to_shapely(bytes(blob)))
    except Exception:
        return None


def _st_length(blob):
    if blob is None:
        return None
    try:
        geom = gaia_to_shapely(bytes(blob))
        # like native GLength: only linear components count (a polygon measures 0)
        if geom.geom_type in ("LineString", "MultiLineString"):
            return geom.length
        if geom.geom_type == "GeometryCollection":
            return sum(g.length for g in geom.geoms if g.geom_type in ("LineString", "MultiLineString"))
        return 0.0
    except Exception:
        return None


def _st_area(blob):
    if blob is None:
        return None
    try:
        return gaia_to_shapely(bytes(blob)).area
    except Exception:
        return None


def _st_centroid(blob):
    if blob is None:
        return None
    try:
        blob = bytes(blob)
        return shapely_to_gaia(gaia_to_shapely(blob).centroid, gaia_srid(blob))
    except Exception:
        return None


def _mbr(blob, index):
    if blob is None:
        return None
    try:
        return gaia_mbr(bytes(blob))[index]
    except Exception:
        return None


def _is_valid(blob):
    if blob is None:
        return -1  # native returns -1 for NULL input
    try:
        return 1 if shapely.is_valid(gaia_to_shapely(bytes(blob))) else 0
    except Exception:
        return -1


def _geometry_type(blob):
    if blob is None:
        return None
    try:
        geom = gaia_to_shapely(bytes(blob))
        name = geom.geom_type.upper()
        if name == "GEOMETRYCOLLECTION":
            return "GEOMETRYCOLLECTION"
        if shapely.has_z(geom):
            return f"{name} Z"
        return name
    except Exception:
        return None


def _spatialite_version():
    # Version of the SpatiaLite dialect this shim is compatible with
    return "5.1.0-aequilibrae-python"


def _hexgrid_base(min_x, min_y, origin_x, origin_y, shift3, shift4, shift):
    """Literal port of libspatialite's get_hexgrid_base (gg_extras.c)."""
    by = origin_y
    odd_even = 0
    while True:
        southward = min_y < origin_y
        if (by <= min_y) if southward else (by >= min_y):
            if odd_even:
                bx = origin_x - shift3 / 2.0 if southward else origin_x + shift3 / 2.0
            else:
                bx = origin_x
            while True:
                if min_x < origin_x:  # going westward
                    if bx - shift4 < min_x:
                        return bx, by, odd_even
                    bx -= shift3
                else:  # going eastward
                    if bx + shift4 > min_x:
                        return bx, by, odd_even
                    bx += shift3
        by = by - shift if southward else by + shift
        odd_even = 0 if odd_even else 1


def _hexagonal_grid(geom_blob, size, mode=0, origin_blob=None):
    """Literal port of libspatialite's gaiaHexagonalGrid (gg_extras.c).

    ``size`` is the hexagon side length; the honeycomb lattice is anchored at
    ``origin`` (default 0/0) and covers the MBR of ``geom``; a hexagon is
    emitted when it intersects ``geom`` (GEOS semantics, same as native).
    """
    if geom_blob is None or size is None or size <= 0:
        return None
    from math import pi, sin

    geom_blob = bytes(geom_blob)
    geom = gaia_to_shapely(geom_blob)
    min_x, min_y, max_x, max_y = geom.bounds
    ox, oy = gaia_point_xy(bytes(origin_blob)) if origin_blob is not None else (0.0, 0.0)

    size = float(size)
    shift = size * sin(pi / 3.0)
    shift2 = size / 2.0
    shift3 = size * 3.0
    shift4 = size * 2.0

    base_x, base_y, odd_even = _hexgrid_base(min_x, min_y, ox, oy, shift3, shift4, shift)
    base_x -= shift3
    base_y -= shift  # note: native does NOT toggle odd_even for this extra row

    hexes = []
    while (base_y - shift) < max_y:
        x1 = base_x - shift3 / 2.0 if odd_even else base_x
        y1 = base_y
        while x1 < max_x:
            hexagon = shapely.Polygon(
                [
                    (x1, y1),
                    (x1 + shift2, y1 - shift),
                    (x1 + shift2 + size, y1 - shift),
                    (x1 + shift4, y1),
                    (x1 + shift2 + size, y1 + shift),
                    (x1 + shift2, y1 + shift),
                ]
            )
            if hexagon.intersects(geom):
                hexes.append(hexagon)
            x1 += shift3
        base_y += shift
        odd_even = 0 if odd_even else 1

    if not hexes:
        return None
    if mode:
        result = shapely.line_merge(shapely.MultiLineString([h.exterior for h in hexes]))
        if result.geom_type == "LineString":
            result = shapely.MultiLineString([result])
    else:
        result = shapely.MultiPolygon(hexes)
    return shapely_to_gaia(result, gaia_srid(geom_blob))


def _geometry_constraints(blob, declared_type, srid):
    """Backs the ``ggi_*``/``ggu_*`` triggers SpatiaLite generates for each geometry column.

    Returns 1 when the geometry matches the declared type and SRID, 0 otherwise.
    A NULL geometry is always acceptable (NOT NULL is enforced separately).
    Header-only check — no geometry parse — as this runs for every row written.
    """
    if blob is None:
        return 1
    try:
        blob = bytes(blob)
        if int(gaia_srid(blob)) != int(srid):
            return 0
        code = gaia_geometry_code(blob) % 1000000  # compressed flavor is type-equivalent
        if isinstance(declared_type, str):
            declared = GEOMETRY_TYPE_CODES.get(declared_type.upper().replace(" Z", ""), -1)
            declared += 1000 if declared_type.upper().endswith(" Z") else 0
        else:
            declared = int(declared_type)
        if declared % 1000 == 0 and declared // 1000 == code // 1000:  # generic GEOMETRY column
            return 1
        return 1 if code == declared else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Connection-bound functions (spatial index maintenance, metadata admin)
# ---------------------------------------------------------------------------


class _SpatialiteShim:
    """Holds the connection reference needed by side-effecting SpatiaLite functions."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # -- spatial index ------------------------------------------------------

    def rtree_align(self, idx_name, pkid, blob):
        if blob is not None:
            minx, miny, maxx, maxy = gaia_mbr(bytes(blob))
            self.conn.execute(
                f'INSERT OR REPLACE INTO "{idx_name}" (pkid, xmin, xmax, ymin, ymax) VALUES (?, ?, ?, ?, ?)',
                (pkid, minx, maxx, miny, maxy),
            )
        return 1

    def create_spatial_index(self, table, column):
        idx = f"idx_{table}_{column}"
        self.conn.execute(f'CREATE VIRTUAL TABLE "{idx}" USING rtree(pkid, xmin, xmax, ymin, ymax)')
        # Trigger bodies identical to the ones mod_spatialite generates, so the
        # database keeps working if it is later opened with the native extension.
        self.conn.execute(
            f'CREATE TRIGGER "gii_{table}_{column}" AFTER INSERT ON "{table}"\n'
            f"FOR EACH ROW BEGIN\n"
            f'DELETE FROM "{idx}" WHERE pkid=NEW.ROWID;\n'
            f"SELECT RTreeAlign('{idx}', NEW.ROWID, NEW.\"{column}\");\n"
            f"END"
        )
        self.conn.execute(
            f'CREATE TRIGGER "giu_{table}_{column}" AFTER UPDATE OF "{column}" ON "{table}"\n'
            f"FOR EACH ROW BEGIN\n"
            f'DELETE FROM "{idx}" WHERE pkid=NEW.ROWID;\n'
            f"SELECT RTreeAlign('{idx}', NEW.ROWID, NEW.\"{column}\");\n"
            f"END"
        )
        self.conn.execute(
            f'CREATE TRIGGER "gid_{table}_{column}" AFTER DELETE ON "{table}"\n'
            f"FOR EACH ROW BEGIN\n"
            f'DELETE FROM "{idx}" WHERE pkid=OLD.ROWID;\n'
            f"END"
        )
        self.conn.execute(
            f'INSERT INTO "{idx}" (pkid, xmin, xmax, ymin, ymax) '
            f'SELECT ROWID, MbrMinX("{column}"), MbrMaxX("{column}"), MbrMinY("{column}"), MbrMaxY("{column}") '
            f'FROM "{table}" WHERE "{column}" IS NOT NULL'
        )
        self.conn.execute(
            "UPDATE geometry_columns SET spatial_index_enabled = 1 "
            "WHERE Lower(f_table_name) = Lower(?) AND Lower(f_geometry_column) = Lower(?)",
            (table, column),
        )
        return 1

    def check_spatial_index(self, table, column):
        idx = f"idx_{table}_{column}"
        try:
            index_pkids = {r[0] for r in self.conn.execute(f'SELECT pkid FROM "{idx}"')}
            table_rowids = {
                r[0] for r in self.conn.execute(f'SELECT ROWID FROM "{table}" WHERE "{column}" IS NOT NULL')
            }
        except sqlite3.OperationalError:
            return None
        return 1 if index_pkids == table_rowids else 0

    def recover_spatial_index(self, table, column):
        idx = f"idx_{table}_{column}"
        try:
            self.conn.execute(f'DELETE FROM "{idx}"')
        except sqlite3.OperationalError:
            return self.create_spatial_index(table, column)
        self.conn.execute(
            f'INSERT INTO "{idx}" (pkid, xmin, xmax, ymin, ymax) '
            f'SELECT ROWID, MbrMinX("{column}"), MbrMaxX("{column}"), MbrMinY("{column}"), MbrMaxY("{column}") '
            f'FROM "{table}" WHERE "{column}" IS NOT NULL'
        )
        return 1

    # -- metadata administration -------------------------------------------

    def _geometry_columns_of(self, table):
        return [
            r[0]
            for r in self.conn.execute(
                "SELECT f_geometry_column FROM geometry_columns WHERE Lower(f_table_name) = Lower(?)", (table,)
            )
        ]

    def add_geometry_column(self, table, column, srid, geom_type, dims="XY", notnull=0):
        code = GEOMETRY_TYPE_CODES[str(geom_type).upper()] + _dims_code(dims)
        coord_dim = 3 if _dims_code(dims) in (1000, 3000) else 2
        self.conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {str(geom_type).upper()}')
        # OR REPLACE: a raw DROP TABLE leaves orphaned metadata behind, which native
        # SpatiaLite tolerates when the geometry column is registered again
        self.conn.execute(
            "INSERT OR REPLACE INTO geometry_columns "
            "(f_table_name, f_geometry_column, geometry_type, coord_dimension, srid, spatial_index_enabled) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (table, column, code, coord_dim, int(srid)),
        )
        for aux, values in (
            ("geometry_columns_auth", "(?, ?, 0, 0)"),
            ("geometry_columns_statistics", "(?, ?, NULL, NULL, NULL, NULL, NULL, NULL)"),
            ("geometry_columns_time", "(?, ?, '0000-01-01T00:00:00.000Z', '0000-01-01T00:00:00.000Z', "
                                      "'0000-01-01T00:00:00.000Z')"),
        ):
            try:
                self.conn.execute(f"INSERT OR IGNORE INTO {aux} VALUES {values}", (table, column))
            except sqlite3.OperationalError:
                pass  # aux table absent in minimally-initialised databases
        self._create_geometry_triggers(table, column)
        return 1

    def _create_geometry_triggers(self, table, column):
        # Same text as mod_spatialite generates (see any AequilibraE project database)
        for prefix, event in (("ggi", "INSERT"), ("ggu", f'UPDATE OF "{column}"')):
            self.conn.execute(
                f'CREATE TRIGGER "{prefix}_{table}_{column}" BEFORE {event} ON "{table}"\n'
                f"FOR EACH ROW BEGIN\n"
                f"SELECT RAISE(ROLLBACK, '{table}.{column} violates Geometry constraint "
                f"[geom-type or SRID not allowed]')\n"
                f"WHERE (SELECT geometry_type FROM geometry_columns\n"
                f"WHERE Lower(f_table_name) = Lower('{table}') AND "
                f"Lower(f_geometry_column) = Lower('{column}')\n"
                f'AND GeometryConstraints(NEW."{column}", geometry_type, srid) = 1) IS NULL;\n'
                f"END"
            )
        if self._has_table("geometry_columns_time"):
            for prefix, event, field in (
                ("tmi", "INSERT", "last_insert"),
                ("tmu", "UPDATE", "last_update"),
                ("tmd", "DELETE", "last_delete"),
            ):
                self.conn.execute(
                    f'CREATE TRIGGER "{prefix}_{table}_{column}" AFTER {event} ON "{table}"\n'
                    f"FOR EACH ROW BEGIN\n"
                    f"UPDATE geometry_columns_time SET {field} = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')\n"
                    f"WHERE Lower(f_table_name) = Lower('{table}') AND "
                    f"Lower(f_geometry_column) = Lower('{column}');\n"
                    f"END"
                )

    def _has_table(self, name) -> bool:
        sql = "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND Lower(name) = Lower(?)"
        return self.conn.execute(sql, (name,)).fetchone() is not None

    def _spatial_index_enabled(self, table, column) -> bool:
        row = self.conn.execute(
            "SELECT spatial_index_enabled FROM geometry_columns "
            "WHERE Lower(f_table_name) = Lower(?) AND Lower(f_geometry_column) = Lower(?)",
            (table, column),
        ).fetchone()
        return bool(row and row[0])

    def get_layer_extent(self, table, column=None):
        columns = [column] if column else self._geometry_columns_of(table)
        if not columns:
            return None
        column = columns[0]
        row = self.conn.execute(
            f'SELECT MIN(MbrMinX("{column}")), MIN(MbrMinY("{column}")), '
            f'MAX(MbrMaxX("{column}")), MAX(MbrMaxY("{column}")), COUNT(*) '
            f'FROM "{table}" WHERE "{column}" IS NOT NULL'
        ).fetchone()
        if not row or not row[4]:
            return None
        minx, miny, maxx, maxy = row[:4]
        srid_row = self.conn.execute(
            "SELECT srid FROM geometry_columns "
            "WHERE Lower(f_table_name) = Lower(?) AND Lower(f_geometry_column) = Lower(?)",
            (table, column),
        ).fetchone()
        srid = srid_row[0] if srid_row else 4326
        return shapely_to_gaia(shapely.box(minx, miny, maxx, maxy), srid)

    def update_layer_statistics(self, table=None, column=None):
        where, args = "", []
        if table:
            where = "WHERE Lower(f_table_name) = Lower(?)"
            args.append(table)
            if column:
                where += " AND Lower(f_geometry_column) = Lower(?)"
                args.append(column)
        for tbl, col in self.conn.execute(
            f"SELECT f_table_name, f_geometry_column FROM geometry_columns {where}", args
        ).fetchall():
            row = self.conn.execute(
                f'SELECT COUNT(*), MIN(MbrMinX("{col}")), MIN(MbrMinY("{col}")), '
                f'MAX(MbrMaxX("{col}")), MAX(MbrMaxY("{col}")) FROM "{tbl}" WHERE "{col}" IS NOT NULL'
            ).fetchone()
            self.conn.execute(
                "UPDATE geometry_columns_statistics SET last_verified = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), "
                "row_count = ?, extent_min_x = ?, extent_min_y = ?, extent_max_x = ?, extent_max_y = ? "
                "WHERE Lower(f_table_name) = Lower(?) AND Lower(f_geometry_column) = Lower(?)",
                (*row, tbl, col),
            )
        return 1

    def invalidate_layer_statistics(self, table=None, column=None):
        where, args = "1=1", []
        if table:
            where = "Lower(f_table_name) = Lower(?)"
            args.append(table)
            if column:
                where += " AND Lower(f_geometry_column) = Lower(?)"
                args.append(column)
        self.conn.execute(
            "UPDATE geometry_columns_statistics SET last_verified = NULL, row_count = NULL, "
            "extent_min_x = NULL, extent_min_y = NULL, extent_max_x = NULL, extent_max_y = NULL "
            f"WHERE {where}",
            args,
        )
        return 1

    def rename_table(self, db_prefix, old, new):
        geo_cols = self._geometry_columns_of(old)
        for col in geo_cols:
            indexed = self._spatial_index_enabled(old, col)
            self._drop_table_triggers(old, col)
            if indexed:
                self.conn.execute(f'ALTER TABLE "idx_{old}_{col}" RENAME TO "idx_{new}_{col}"')
        self.conn.execute(f'ALTER TABLE "{old}" RENAME TO "{new}"')
        for meta in (
            "geometry_columns",
            "geometry_columns_auth",
            "geometry_columns_statistics",
            "geometry_columns_time",
        ):
            if self._has_table(meta):
                self.conn.execute(
                    f"UPDATE {meta} SET f_table_name = ? WHERE Lower(f_table_name) = Lower(?)", (new, old)
                )
        for col in geo_cols:
            self._create_geometry_triggers(new, col)
            if self._spatial_index_enabled(new, col):
                self._create_index_triggers(new, col)
        return 1

    def _create_index_triggers(self, table, column):
        idx = f"idx_{table}_{column}"
        self.conn.execute(
            f'CREATE TRIGGER "gii_{table}_{column}" AFTER INSERT ON "{table}"\n'
            f"FOR EACH ROW BEGIN\n"
            f'DELETE FROM "{idx}" WHERE pkid=NEW.ROWID;\n'
            f"SELECT RTreeAlign('{idx}', NEW.ROWID, NEW.\"{column}\");\n"
            f"END"
        )
        self.conn.execute(
            f'CREATE TRIGGER "giu_{table}_{column}" AFTER UPDATE OF "{column}" ON "{table}"\n'
            f"FOR EACH ROW BEGIN\n"
            f'DELETE FROM "{idx}" WHERE pkid=NEW.ROWID;\n'
            f"SELECT RTreeAlign('{idx}', NEW.ROWID, NEW.\"{column}\");\n"
            f"END"
        )
        self.conn.execute(
            f'CREATE TRIGGER "gid_{table}_{column}" AFTER DELETE ON "{table}"\n'
            f"FOR EACH ROW BEGIN\n"
            f'DELETE FROM "{idx}" WHERE pkid=OLD.ROWID;\n'
            f"END"
        )

    def _drop_table_triggers(self, table, column):
        for prefix in ("ggi", "ggu", "gii", "giu", "gid", "tmi", "tmu", "tmd"):
            self.conn.execute(f'DROP TRIGGER IF EXISTS "{prefix}_{table}_{column}"')

    def drop_table(self, db_prefix, table):
        for col in self._geometry_columns_of(table):
            self.conn.execute(f'DROP TABLE IF EXISTS "idx_{table}_{col}"')
        self.conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        for meta in (
            "geometry_columns",
            "geometry_columns_auth",
            "geometry_columns_statistics",
            "geometry_columns_time",
        ):
            if self._has_table(meta):
                self.conn.execute(f"DELETE FROM {meta} WHERE Lower(f_table_name) = Lower(?)", (table,))
        return 1

    def init_spatial_metadata(self, *args):
        """Minimal InitSpatialMetaData: enough for AddGeometryColumn/CreateSpatialIndex to work."""
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS spatial_ref_sys (srid INTEGER NOT NULL PRIMARY KEY, "
            "auth_name TEXT NOT NULL, auth_srid INTEGER NOT NULL, ref_sys_name TEXT NOT NULL DEFAULT 'Unknown', "
            "proj4text TEXT NOT NULL, srtext TEXT NOT NULL DEFAULT 'Undefined')"
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO spatial_ref_sys (srid, auth_name, auth_srid, ref_sys_name, proj4text, srtext) "
            "VALUES (4326, 'epsg', 4326, 'WGS 84', '+proj=longlat +datum=WGS84 +no_defs', "
            "'GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],"
            "PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]')"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS geometry_columns (f_table_name TEXT NOT NULL, "
            "f_geometry_column TEXT NOT NULL, geometry_type INTEGER NOT NULL, coord_dimension INTEGER NOT NULL, "
            "srid INTEGER NOT NULL, spatial_index_enabled INTEGER NOT NULL, "
            "CONSTRAINT pk_geom_cols PRIMARY KEY (f_table_name, f_geometry_column))"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS geometry_columns_statistics (f_table_name TEXT NOT NULL, "
            "f_geometry_column TEXT NOT NULL, last_verified TIMESTAMP, row_count INTEGER, "
            "extent_min_x DOUBLE, extent_min_y DOUBLE, extent_max_x DOUBLE, extent_max_y DOUBLE, "
            "CONSTRAINT pk_gc_statistics PRIMARY KEY (f_table_name, f_geometry_column))"
        )
        return 1


# ---------------------------------------------------------------------------
# Legacy trigger rewriting: SpatialIndex virtual table -> direct R*Tree query
# ---------------------------------------------------------------------------

_SPATIALINDEX_RE = re.compile(
    r"SELECT\s+ROWID\s+FROM\s+SpatialIndex\s+WHERE\s+f_table_name\s*=\s*'(\w+)'\s+AND\s+search_frame\s*=\s*",
    re.IGNORECASE,
)


def _find_balanced_expr(sql: str, start: int) -> int:
    """Return the end index of the expression starting at ``start``.

    The expression ends at the closing parenthesis of the enclosing subquery,
    i.e. the first ``)`` not opened within the expression itself.
    """
    depth = 0
    for i in range(start, len(sql)):
        c = sql[i]
        if c == "(":
            depth += 1
        elif c == ")":
            if depth == 0:
                return i
            depth -= 1
    return len(sql)


def rewrite_spatialindex_sql(sql: str):
    """Rewrite ``SELECT ROWID FROM SpatialIndex WHERE f_table_name='t' AND search_frame=E``
    into a direct query on the ``idx_t_geometry`` R*Tree table. Returns None when
    the statement contains no such pattern."""
    out, pos, changed = [], 0, False
    while True:
        m = _SPATIALINDEX_RE.search(sql, pos)
        if m is None:
            out.append(sql[pos:])
            break
        changed = True
        table = m.group(1)
        expr_start = m.end()
        expr_end = _find_balanced_expr(sql, expr_start)
        expr = sql[expr_start:expr_end].strip()
        out.append(sql[pos : m.start()])
        out.append(
            f'SELECT pkid FROM "idx_{table}_geometry" WHERE '
            f"xmin <= MbrMaxX({expr}) AND xmax >= MbrMinX({expr}) AND "
            f"ymin <= MbrMaxY({expr}) AND ymax >= MbrMinY({expr})"
        )
        pos = expr_end
    return "".join(out) if changed else None


def upgrade_legacy_spatialindex_triggers(conn: sqlite3.Connection) -> int:
    """Rewrite triggers in pre-existing databases that query the SpatialIndex
    virtual table (which only exists when the native extension is loaded).

    Returns the number of triggers rewritten. Silently skips read-only databases —
    triggers only fire on writes, which such connections cannot perform anyway.
    """
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND sql LIKE '%SpatialIndex%'"
        ).fetchall()
    except sqlite3.DatabaseError:
        return 0
    rewritten = 0
    for name, sql in rows:
        new_sql = rewrite_spatialindex_sql(sql)
        if new_sql is None:
            continue
        try:
            conn.execute(f'DROP TRIGGER "{name}"')
            conn.execute(new_sql)
            rewritten += 1
        except sqlite3.OperationalError:
            return rewritten  # read-only database
    return rewritten


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_spatialite_functions(conn: sqlite3.Connection) -> None:
    """Register the SpatiaLite-compatible SQL functions AequilibraE uses on ``conn``."""
    det = {"deterministic": True}

    # constructors
    for name in ("GeomFromWKB", "ST_GeomFromWKB", "GeometryFromWKB"):
        conn.create_function(name, 2, _geom_from_wkb, **det)
        conn.create_function(name, 1, _geom_from_wkb, **det)
    for name in ("GeomFromText", "ST_GeomFromText", "GeometryFromText"):
        conn.create_function(name, 2, _geom_from_text, **det)
        conn.create_function(name, 1, _geom_from_text, **det)
    conn.create_function("MakePoint", 3, _make_point, **det)
    conn.create_function("MakePoint", 2, _make_point, **det)
    conn.create_function("MakeLine", 2, _make_line, **det)
    for name in ("ST_Multi", "CastToMulti"):
        conn.create_function(name, 1, _st_multi, **det)

    # serializers / accessors
    for name in ("AsBinary", "ST_AsBinary"):
        conn.create_function(name, 1, _as_binary, **det)
    for name in ("AsText", "ST_AsText", "AsWKT"):
        conn.create_function(name, 1, _as_text, **det)
    for name in ("X", "ST_X"):
        conn.create_function(name, 1, _st_x, **det)
    for name in ("Y", "ST_Y"):
        conn.create_function(name, 1, _st_y, **det)
    for name in ("Srid", "ST_SRID"):
        conn.create_function(name, 1, _srid, **det)
    conn.create_function("GeometryType", 1, _geometry_type, **det)
    conn.create_function("IsValid", 1, _is_valid, **det)

    # linear referencing
    for name in ("StartPoint", "ST_StartPoint"):
        conn.create_function(name, 1, _start_point, **det)
    for name in ("EndPoint", "ST_EndPoint"):
        conn.create_function(name, 1, _end_point, **det)
    conn.create_function("SetStartPoint", 2, _set_start_point, **det)
    conn.create_function("SetEndPoint", 2, _set_end_point, **det)

    # measurement
    conn.create_function("GeodesicLength", 1, _geodesic_length, **det)
    for name in ("GLength", "ST_Length"):
        conn.create_function(name, 1, _st_length, **det)
    for name in ("Area", "ST_Area"):
        conn.create_function(name, 1, _st_area, **det)
    for name in ("Centroid", "ST_Centroid"):
        conn.create_function(name, 1, _st_centroid, **det)

    # MBR accessors (used heavily by the rewritten spatial-index queries)
    for i, name in enumerate(("MbrMinX", "MbrMinY", "MbrMaxX", "MbrMaxY")):
        conn.create_function(name, 1, (lambda idx: lambda blob: _mbr(blob, idx))(i), **det)

    # constraint checking (used by the ggi_*/ggu_* triggers)
    conn.create_function("GeometryConstraints", 3, _geometry_constraints, **det)

    conn.create_function("spatialite_version", 0, _spatialite_version, **det)
    for name in ("HexagonalGrid", "ST_HexagonalGrid"):
        conn.create_function(name, 2, _hexagonal_grid, **det)
        conn.create_function(name, 3, _hexagonal_grid, **det)
        conn.create_function(name, 4, _hexagonal_grid, **det)

    # connection-bound admin functions
    shim = _SpatialiteShim(conn)
    conn.create_function("RTreeAlign", 3, shim.rtree_align)
    conn.create_function("CreateSpatialIndex", 2, shim.create_spatial_index)
    conn.create_function("CheckSpatialIndex", 2, shim.check_spatial_index)
    conn.create_function("RecoverSpatialIndex", 2, shim.recover_spatial_index)
    conn.create_function("AddGeometryColumn", 5, shim.add_geometry_column)
    conn.create_function("AddGeometryColumn", 6, shim.add_geometry_column)
    conn.create_function("GetLayerExtent", 1, shim.get_layer_extent)
    conn.create_function("GetLayerExtent", 2, shim.get_layer_extent)
    conn.create_function("UpdateLayerStatistics", 0, shim.update_layer_statistics)
    conn.create_function("UpdateLayerStatistics", 1, shim.update_layer_statistics)
    conn.create_function("UpdateLayerStatistics", 2, shim.update_layer_statistics)
    conn.create_function("InvalidateLayerStatistics", 0, shim.invalidate_layer_statistics)
    conn.create_function("InvalidateLayerStatistics", 1, shim.invalidate_layer_statistics)
    conn.create_function("InvalidateLayerStatistics", 2, shim.invalidate_layer_statistics)
    conn.create_function("RenameTable", 3, shim.rename_table)
    conn.create_function("DropTable", 2, shim.drop_table)
    conn.create_function("DropGeoTable", 2, shim.drop_table)
    conn.create_function("InitSpatialMetaData", -1, shim.init_spatial_metadata)
