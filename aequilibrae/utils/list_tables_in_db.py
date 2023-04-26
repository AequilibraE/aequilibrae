def list_tables_in_db(conn):
    """
    Return a list with all tables within a database.

    :Arguments:
         **conn** (:obj: `sqlite3.Connection`): database connection
    """
    return [x[0] for x in conn.execute("SELECT name FROM sqlite_master WHERE type ='table'").fetchall()]
