"""Pure-Python codec for SpatiaLite's internal geometry BLOB format ("Gaia" format).

SpatiaLite stores geometries in its own BLOB layout rather than plain WKB:

    byte 0        0x00                        BLOB start marker
    byte 1        endianness                  0x01 little-endian, 0x00 big-endian
    bytes 2-5     SRID                        int32
    bytes 6-37    MBR                         minx, miny, maxx, maxy as doubles
    byte 38       0x7C                        MBR end marker
    bytes 39-42   geometry class type         int32, same codes as WKB
    bytes 43..    geometry payload            WKB-like, but sub-geometries of
                                              multi/collection types are prefixed
                                              with the marker 0x69 + int32 type
                                              instead of WKB's endian byte + type
    last byte     0xFE                        BLOB end marker

This module converts between that format and shapely geometries without any
native SpatiaLite library. It also reads (but never writes) the optional
"TinyPoint" compact encoding introduced in SpatiaLite 5.

References: https://www.gaia-gis.it/gaia-sins/BLOB-Geometry.html
"""

import struct

import shapely
from shapely.geometry.base import BaseGeometry

GAIA_START = 0x00
GAIA_END = 0xFE
GAIA_MBR_END = 0x7C
GAIA_ENTITY = 0x69
TINYPOINT_START = 0x01

# WKB geometry type codes (2D base codes)
POINT, LINESTRING, POLYGON, MULTIPOINT, MULTILINESTRING, MULTIPOLYGON, COLLECTION = range(1, 8)

_MULTI_TYPES = {MULTIPOINT, MULTILINESTRING, MULTIPOLYGON, COLLECTION}


class GaiaError(ValueError):
    pass


def _dims_from_type(gtype: int):
    """Return (base_type, has_z, has_m, compressed) from a gaia/WKB geometry class code.

    Compressed classes (base + 1000000) store intermediate linestring/ring vertices
    as float32 deltas; they are read but never written by this module.
    """
    compressed = gtype >= 1000000
    base = gtype % 1000
    flavor = (gtype // 1000) % 1000
    if flavor == 0:
        return base, False, False, compressed
    if flavor == 1:  # XYZ
        return base, True, False, compressed
    if flavor == 2:  # XYM
        return base, False, True, compressed
    if flavor == 3:  # XYZM
        return base, True, True, compressed
    raise GaiaError(f"Unknown geometry type code {gtype}")


def _point_size(has_z: bool, has_m: bool) -> int:
    return 8 * (2 + int(has_z) + int(has_m))


def gaia_to_wkb(blob: bytes) -> bytes:
    """Convert a SpatiaLite geometry BLOB to standard WKB (ignoring any M values)."""
    if blob is None or len(blob) < 5:
        raise GaiaError("Not a SpatiaLite geometry BLOB")

    if blob[0] == TINYPOINT_START:
        return _tinypoint_to_wkb(blob)

    if blob[0] != GAIA_START or blob[-1] != GAIA_END or len(blob) < 45 or blob[38] != GAIA_MBR_END:
        raise GaiaError("Not a SpatiaLite geometry BLOB")

    endian = blob[1]
    if endian not in (0, 1):
        raise GaiaError("Invalid endianness marker in SpatiaLite BLOB")
    fmt = "<" if endian == 1 else ">"
    endian_byte = bytes([endian])

    (gtype,) = struct.unpack_from(f"{fmt}i", blob, 39)
    base, has_z, has_m, compressed = _dims_from_type(gtype)

    out = bytearray()
    out += endian_byte
    out += struct.pack(f"{fmt}I", gtype % 1000000)  # WKB has no "compressed" flavor

    pos = 43
    if base in (POINT, LINESTRING, POLYGON):
        pos = _copy_simple_payload(blob, pos, out, fmt, base, has_z, has_m, compressed)
    elif base in _MULTI_TYPES:
        (n_items,) = struct.unpack_from(f"{fmt}i", blob, pos)
        pos += 4
        out += struct.pack(f"{fmt}I", n_items)
        for _ in range(n_items):
            if blob[pos] != GAIA_ENTITY:
                raise GaiaError("Malformed SpatiaLite BLOB: missing entity marker")
            pos += 1
            (sub_type,) = struct.unpack_from(f"{fmt}i", blob, pos)
            pos += 4
            sub_base, sub_z, sub_m, sub_comp = _dims_from_type(sub_type)
            out += endian_byte
            out += struct.pack(f"{fmt}I", sub_type % 1000000)
            pos = _copy_simple_payload(blob, pos, out, fmt, sub_base, sub_z, sub_m, sub_comp)
    else:
        raise GaiaError(f"Unknown geometry class {base}")

    if pos != len(blob) - 1:
        raise GaiaError("Malformed SpatiaLite BLOB: trailing bytes")
    return bytes(out)


def _copy_simple_payload(blob, pos, out, fmt, base, has_z, has_m, compressed=False) -> int:
    """Copy the payload of a point/linestring/polygon (identical layout in gaia and WKB),
    expanding compressed vertex sequences to plain doubles."""
    psize = _point_size(has_z, has_m)
    if base == POINT:  # points are never compressed
        out += blob[pos : pos + psize]
        return pos + psize
    if base == LINESTRING:
        (n_pts,) = struct.unpack_from(f"{fmt}i", blob, pos)
        if compressed:
            return _expand_compressed_sequence(blob, pos, out, fmt, has_z, has_m)
        end = pos + 4 + n_pts * psize
        out += blob[pos:end]
        return end
    if base == POLYGON:
        (n_rings,) = struct.unpack_from(f"{fmt}i", blob, pos)
        out += blob[pos : pos + 4]
        pos += 4
        for _ in range(n_rings):
            if compressed:
                pos = _expand_compressed_sequence(blob, pos, out, fmt, has_z, has_m)
            else:
                (n_pts,) = struct.unpack_from(f"{fmt}i", blob, pos)
                end = pos + 4 + n_pts * psize
                out += blob[pos:end]
                pos = end
        return pos
    raise GaiaError(f"Not a simple geometry class: {base}")


def _expand_compressed_sequence(blob, pos, out, fmt, has_z, has_m) -> int:
    """Expand one compressed vertex sequence (linestring body or polygon ring) to doubles.

    Layout (per SpatiaLite's gg_gaia.c): first and last vertices as full doubles;
    every intermediate vertex as float32 deltas from the previous vertex. M values,
    when present, are always stored as doubles and never delta-encoded.
    """
    (n_pts,) = struct.unpack_from(f"{fmt}i", blob, pos)
    pos += 4
    out += struct.pack(f"{fmt}i", n_pts)
    n_dims = 2 + int(has_z)
    prev = None
    for i in range(n_pts):
        if i == 0 or i == n_pts - 1:
            coords = list(struct.unpack_from(f"{fmt}{n_dims}d", blob, pos))
            pos += 8 * n_dims
        else:
            deltas = struct.unpack_from(f"{fmt}{n_dims}f", blob, pos)
            pos += 4 * n_dims
            coords = [p + d for p, d in zip(prev, deltas, strict=False)]
        m = ()
        if has_m:
            m = struct.unpack_from(f"{fmt}d", blob, pos)
            pos += 8
        out += struct.pack(f"{fmt}{n_dims}d", *coords)
        if has_m:
            out += struct.pack(f"{fmt}d", *m)
        prev = coords
    return pos


def _tinypoint_to_wkb(blob: bytes) -> bytes:
    """Convert a SpatiaLite 5 TinyPoint BLOB to WKB (read-only support)."""
    # Layout: 0x01 | endian | srid int32 | type byte | x double | y double [| z ][| m ] | 0xFE
    if len(blob) < 24 or blob[-1] != GAIA_END:
        raise GaiaError("Not a SpatiaLite TinyPoint BLOB")
    endian = blob[1]
    fmt = "<" if endian == 1 else ">"
    tp_type = blob[6]  # 1=XY, 2=XYZ, 3=XYM, 4=XYZM
    x, y = struct.unpack_from(f"{fmt}dd", blob, 7)
    if tp_type in (2, 4):
        (z,) = struct.unpack_from(f"{fmt}d", blob, 23)
        return bytes([endian]) + struct.pack(f"{fmt}Iddd", 1001, x, y, z)
    return bytes([endian]) + struct.pack(f"{fmt}Idd", 1, x, y)


def gaia_geometry_code(blob: bytes) -> int:
    """Read the geometry class code straight from the BLOB header (no parse)."""
    if blob is None or len(blob) < 43:
        raise GaiaError("Not a SpatiaLite geometry BLOB")
    if blob[0] == TINYPOINT_START:
        return POINT + (1000 if blob[6] in (2, 4) else 0)
    fmt = "<" if blob[1] == 1 else ">"
    return struct.unpack_from(f"{fmt}i", blob, 39)[0]


def linestring_lonlats(blob: bytes):
    """Return (lons, lats) of an uncompressed LINESTRING BLOB (fast path, no shapely).

    Returns None when the geometry is anything else; callers fall back to full decoding.
    """
    if blob is None or len(blob) < 47 or blob[0] != GAIA_START:
        return None
    fmt = "<" if blob[1] == 1 else ">"
    (gtype,) = struct.unpack_from(f"{fmt}i", blob, 39)
    base, has_z, has_m, compressed = _dims_from_type(gtype)
    if base != LINESTRING or compressed:
        return None
    (n_pts,) = struct.unpack_from(f"{fmt}i", blob, 43)
    step = _point_size(has_z, has_m) // 8
    coords = struct.unpack_from(f"{fmt}{n_pts * step}d", blob, 47)
    return coords[::step], coords[1::step]


def gaia_srid(blob: bytes) -> int:
    """Extract the SRID from a SpatiaLite geometry BLOB."""
    if blob is None or len(blob) < 6 or blob[0] not in (GAIA_START, TINYPOINT_START):
        raise GaiaError("Not a SpatiaLite geometry BLOB")
    fmt = "<" if blob[1] == 1 else ">"
    (srid,) = struct.unpack_from(f"{fmt}i", blob, 2)
    return srid


def gaia_to_shapely(blob: bytes) -> BaseGeometry:
    return shapely.from_wkb(gaia_to_wkb(blob))


def wkb_to_gaia(wkb: bytes, srid: int = 4326) -> bytes:
    """Convert standard WKB to a SpatiaLite geometry BLOB."""
    return shapely_to_gaia(shapely.from_wkb(wkb), srid)


def shapely_to_gaia(geom: BaseGeometry, srid: int = 4326) -> bytes:
    """Serialize a shapely geometry as a SpatiaLite BLOB (little-endian, uncompressed)."""
    if geom is None or geom.is_empty:
        raise GaiaError("Cannot serialize empty geometry to SpatiaLite BLOB")

    # little-endian ISO WKB: gaia uses ISO type codes (1001 for POINT Z), not EWKB flag bits
    wkb = shapely.to_wkb(geom, byte_order=1, flavor="iso")
    minx, miny, maxx, maxy = geom.bounds

    out = bytearray()
    out += struct.pack("<BB", GAIA_START, 1)
    out += struct.pack("<i", int(srid))
    out += struct.pack("<dddd", minx, miny, maxx, maxy)
    out.append(GAIA_MBR_END)

    (gtype,) = struct.unpack_from("<I", wkb, 1)
    base, has_z, has_m, _ = _dims_from_type(gtype)
    out += struct.pack("<i", gtype)

    pos = 5
    if base in (POINT, LINESTRING, POLYGON):
        out += wkb[pos:]
    elif base in _MULTI_TYPES:
        (n_items,) = struct.unpack_from("<i", wkb, pos)
        pos += 4
        out += struct.pack("<i", n_items)
        for _ in range(n_items):
            # each WKB sub-geometry: endian byte + type; gaia: 0x69 marker + type
            (sub_type,) = struct.unpack_from("<I", wkb, pos + 1)
            sub_base, sub_z, sub_m, _ = _dims_from_type(sub_type)
            out.append(GAIA_ENTITY)
            out += struct.pack("<i", sub_type)
            pos += 5
            scratch = bytearray()
            pos = _copy_simple_payload(wkb, pos, scratch, "<", sub_base, sub_z, sub_m)
            out += scratch
    else:
        raise GaiaError(f"Unknown geometry class {base}")

    out.append(GAIA_END)
    return bytes(out)


def make_point_blob(x: float, y: float, srid: int, z: float = None) -> bytes:
    """Build a POINT (or POINT Z) BLOB directly (fast path, no shapely)."""
    header = struct.pack("<BBi", GAIA_START, 1, int(srid)) + struct.pack("<dddd", x, y, x, y)
    if z is None:
        return header + struct.pack("<BiddB", GAIA_MBR_END, POINT, x, y, GAIA_END)
    return header + struct.pack("<BidddB", GAIA_MBR_END, POINT + 1000, x, y, z, GAIA_END)


def gaia_mbr(blob: bytes):
    """Read (minx, miny, maxx, maxy) straight from the BLOB header (no geometry parse)."""
    if blob is None or len(blob) < 24:
        raise GaiaError("Not a SpatiaLite geometry BLOB")
    if blob[0] == TINYPOINT_START:
        fmt = "<" if blob[1] == 1 else ">"
        x, y = struct.unpack_from(f"{fmt}dd", blob, 7)
        return x, y, x, y
    if blob[0] != GAIA_START or len(blob) < 45:
        raise GaiaError("Not a SpatiaLite geometry BLOB")
    fmt = "<" if blob[1] == 1 else ">"
    minx, miny, maxx, maxy = struct.unpack_from(f"{fmt}dddd", blob, 6)
    return minx, miny, maxx, maxy


def gaia_point_xy(blob: bytes):
    """Return (x, y) of a POINT BLOB (fast path)."""
    if blob is None:
        raise GaiaError("Not a SpatiaLite geometry BLOB")
    if blob[0] == TINYPOINT_START:
        fmt = "<" if blob[1] == 1 else ">"
        return struct.unpack_from(f"{fmt}dd", blob, 7)
    fmt = "<" if blob[1] == 1 else ">"
    (gtype,) = struct.unpack_from(f"{fmt}i", blob, 39)
    base, _, _, _ = _dims_from_type(gtype)
    if base != POINT:
        raise GaiaError("Not a POINT geometry")
    return struct.unpack_from(f"{fmt}dd", blob, 43)


def linestring_boundary_point(blob: bytes, start: bool):
    """Return the first/last vertex of a LINESTRING BLOB as a POINT BLOB (fast path).

    Returns None for non-linestring input, mirroring SpatiaLite's StartPoint/EndPoint.
    """
    if blob is None or blob[0] != GAIA_START or len(blob) < 45:
        return None
    fmt = "<" if blob[1] == 1 else ">"
    (gtype,) = struct.unpack_from(f"{fmt}i", blob, 39)
    try:
        base, has_z, has_m, compressed = _dims_from_type(gtype)
    except GaiaError:
        return None
    if base != LINESTRING:
        return None
    (n_pts,) = struct.unpack_from(f"{fmt}i", blob, 43)
    if n_pts < 1:
        return None
    if compressed:
        # first and last vertices of a compressed sequence are stored as full doubles
        n_dims = 2 + int(has_z)
        m_size = 8 if has_m else 0
        if start or n_pts == 1:
            offset = 47
        else:
            offset = 47 + (8 * n_dims + m_size) + (n_pts - 2) * (4 * n_dims + m_size)
    else:
        psize = _point_size(has_z, has_m)
        offset = 47 if start else 47 + (n_pts - 1) * psize
    x, y = struct.unpack_from(f"{fmt}dd", blob, offset)
    z = struct.unpack_from(f"{fmt}d", blob, offset + 16)[0] if has_z else None
    return make_point_blob(x, y, gaia_srid(blob), z)


def is_gaia_blob(value) -> bool:
    """Cheap check that a value looks like a SpatiaLite geometry BLOB."""
    return (
        isinstance(value, (bytes, memoryview, bytearray))
        and len(value) >= 24
        and value[0] in (GAIA_START, TINYPOINT_START)
        and value[-1] == GAIA_END
    )
