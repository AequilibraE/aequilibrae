import logging
from contextlib import closing

from aequilibrae.transit.constants import AGENCY_MULTIPLIER
from aequilibrae.transit.functions.db_utils import list_tables_in_db
from aequilibrae.transit.functions.get_srid import get_srid
from aequilibrae.transit.functions.transit_connection import transit_connection


def create_raw_shapes(agency_id: int, select_patterns):
    """
    Adds all shapes provided in the GTFS feed.

    Args:
        *agency_id* (:obj:`int`): agency_id number
        *select_patterns* (:obj:`dict`): dictionary containing patterns.

    """
    logger = logging.getLogger("aequilibrae")
    logger.info(f"Creating transit raw shapes for agency ID: {agency_id}")
    srid = get_srid()

    with closing(transit_connection()) as conn:
        table_list = list_tables_in_db(conn)
        if "raw_shapes" not in table_list:
            conn.execute('CREATE TABLE IF NOT EXISTS "raw_shapes" ("pattern_id"	TEXT, "route_id" TEXT);')
            conn.execute(f'SELECT AddGeometryColumn( "raw_shapes", "geo", {srid}, "LINESTRING", "XY");')
            conn.execute('SELECT CreateSpatialIndex("Link" , "geo");')
        else:
            bottom = agency_id * AGENCY_MULTIPLIER
            top = bottom + AGENCY_MULTIPLIER
            conn.execute("Delete from raw_shapes where pattern_id>=? and pattern_id<?", [bottom, top])
        conn.commit()
        sql = "INSERT into raw_shapes(pattern_id, route_id, geo) VALUES(?,?, GeomFromWKB(?, ?));"
        for pat in select_patterns.values():
            if pat.raw_shape:
                conn.execute(sql, [pat.pattern_id, pat.route_id, pat.raw_shape.wkb, srid])
            else:
                conn.execute(sql, [pat.pattern_id, pat.route_id, pat._stop_based_shape.wkb, srid])
        conn.commit()
        logger.info("   Finished creating raw shapes")
