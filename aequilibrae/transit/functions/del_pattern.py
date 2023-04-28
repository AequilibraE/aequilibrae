from aequilibrae.project.database_connection import database_connection
from aequilibrae.utils.db_utils import commit_and_close


def delete_pattern(pattern_id: int):
    """Deletes all information regarding one specific transit_pattern.

    :Arguments:
        **pattern_id** (:obj:`str`): pattern_id as present in the database
    """
    sqls = [
        """DELETE from trips where trip_id IN
                (select trip_id from trips where pattern_id=?)""",
        "DELETE from trips where pattern_id=?",
        "DELETE from links where pattern_id=?",
        "DELETE from pattern_links where pattern_id=?",
        "DELETE from pattern_mapping where pattern_id=?",
    ]

    with commit_and_close(database_connection("transit")) as conn:
        for sql in sqls:
            conn.execute(sql, [pattern_id])
