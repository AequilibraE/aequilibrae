def migrate(conn):
    conn.execute(
        """
    CREATE TABLE comments (
        id INTEGER PRIMARY KEY,
        post_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (post_id) REFERENCES posts(id),
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """
    )

    conn.execute("CREATE INDEX idx_comments_post_id ON comments(post_id)")
    conn.execute("CREATE INDEX idx_comments_user_id ON comments(user_id)")
