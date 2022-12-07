import logging
from contextlib import closing
from sqlite3 import Cursor

from aequilibrae.transit.constants import AGENCY_MULTIPLIER
from aequilibrae.transit.functions.db_utils import list_tables_in_db
from aequilibrae.transit.functions.get_srid import get_srid
from aequilibrae.transit.functions.transit_connection import transit_connection


def create_raw_shapes(agency_id: int, select_patterns):
    # logger = logging.getLogger("aequilibrae")
    # logger.info(f"Creating transit raw shapes for agency ID: {agency_id}")
    srid = get_srid()

    with closing(transit_connection()) as conn:
        table_list = list_tables_in_db(conn)
        if "transit_raw_shapes" not in table_list:
            conn.execute('CREATE TABLE IF NOT EXISTS "TRANSIT_RAW_SHAPES" ("pattern_id"	TEXT, "route_id" TEXT);')
            conn.execute(f'SELECT AddGeometryColumn( "TRANSIT_RAW_SHAPES", "geo", {srid}, "LINESTRING", "XY");')
            conn.execute('SELECT CreateSpatialIndex("Link" , "geo");')
        else:
            bottom = agency_id * AGENCY_MULTIPLIER
            top = bottom + AGENCY_MULTIPLIER
            conn.execute("Delete from TRANSIT_RAW_SHAPES where pattern_id>=? and pattern_id<?", [bottom, top])
        conn.commit()
        sql = "INSERT into Transit_raw_shapes(pattern_id, route_id, geo) VALUES(?,?, GeomFromWKB(?, ?));"
        for pat in select_patterns.values():  # type: Pattern
            if pat.raw_shape:
                conn.execute(sql, [pat.pattern_id, pat.route_id, pat.raw_shape.wkb, srid])
            else:
                conn.execute(sql, [pat.pattern_id, pat.route_id, pat._stop_based_shape.wkb, srid])
        conn.commit()
        # logger.info("   Finished creating raw shapes")
