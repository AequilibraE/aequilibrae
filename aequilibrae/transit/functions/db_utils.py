def list_tables_in_db(cnx):
    """
    Return a list with all tables within a database.

    Args:
         *cnx* (:obj: `sqlite3.Connection`): database connection
    """
    return [x[0] for x in cnx.execute("SELECT name FROM sqlite_master WHERE type ='table'").fetchall()]
