from polarislib.network.database_connection import supply_database_connection


def delete_pattern(pattern_id: int):
    """Deletes all information regarding one specific transit_pattern

    Args:
        *pattern_id* (:obj:`str`): pattern_id as present in the database
    """
    sqls = [
        """DELETE from transit_trips_schedule where trip_id IN
                (select trip_id from transit_trips where pattern_id=?)""",
        "DELETE from Transit_Trips where pattern_id=?",
        "DELETE from Transit_Links where pattern_id=?",
        "DELETE from Transit_Pattern_Links where pattern_id=?",
        "DELETE from Transit_Pattern_Mapping where pattern_id=?",
        "DELETE from Transit_Patterns where pattern_id=?",
    ]

    sql_cleaning = "DELETE from transit_routes where route_id not in (select route_id from transit_patterns)"

    conn = supply_database_connection()
    for sql in sqls:
        conn.execute(sql, [pattern_id])
    conn.execute(sql_cleaning)
    conn.commit()
    conn.close()
