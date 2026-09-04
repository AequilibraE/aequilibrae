def migrate(*, project_conn, transit_conn=None, results_conn=None):
    project_conn.execute(
        """CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    project_conn.execute("INSERT INTO users (username, email) VALUES ('admin', 'admin@example.com')")
