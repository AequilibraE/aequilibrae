from aequilibrae.transit.functions.transit_connection import transit_connection


def delete_pattern(pattern_id: int):
    """Deletes all information regarding one specific transit_pattern.

    Args:
        *pattern_id* (:obj:`str`): pattern_id as present in the database
    """
    sqls = [
        """DELETE from trips where trip_id IN
                (select trip_id from trips where pattern_id=?)""",
        "DELETE from trips where pattern_id=?",
        "DELETE from links where pattern_id=?",
        "DELETE from pattern_links where pattern_id=?",
        "DELETE from pattern_mapping where pattern_id=?",
    ]

    conn = transit_connection()
    for sql in sqls:
        conn.execute(sql, [pattern_id])
    conn.commit()
    conn.close()
