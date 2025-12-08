# This file doesn't have a migrate function
def wrong_function_name(conn):
    conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
