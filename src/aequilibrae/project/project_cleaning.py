def clean(project):
    # SQLite does not define trigger execution order, so remove extraneous
    # non-centroid nodes at project lifecycle boundaries.
    with project.db_connection as conn:
        conn.execute(
            """DELETE FROM nodes WHERE is_centroid=0 AND
            (SELECT count(*) FROM links WHERE a_node=node_id OR b_node=node_id)=0"""
        )
