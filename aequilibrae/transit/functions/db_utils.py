def list_tables_in_db(cnx):
    return [x[0] for x in cnx.execute("SELECT name FROM sqlite_master WHERE type ='table'").fetchall()]
