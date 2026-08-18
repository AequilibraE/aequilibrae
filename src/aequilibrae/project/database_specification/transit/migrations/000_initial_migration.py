def migrate(*, project_conn, transit_conn=None, results_conn=None):
    transit_conn.execute(
        """CREATE TABLE IF NOT EXISTS migrations (
            id INTEGER PRIMARY KEY CHECK( id >= 0),
            name TEXT NOT NULL,
            status TEXT DEFAULT 'MISSING' CHECK( status IN ('APPLIED', 'SKIPPED', 'MISSING') ) NOT NULL,
            date TIMESTAMP DEFAULT NULL
        )"""
    )
