def migrate(*, project_conn, transit_conn=None, results_conn=None):
    project_conn.execute(
        """CREATE TABLE posts (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )"""
    )
    project_conn.execute("CREATE INDEX idx_posts_user_id ON posts(user_id)")
