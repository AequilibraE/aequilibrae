def clean(project):
    # Since we cannot decide the order of trigger execution in SQLITE, we make sure to remove any
    # extraneous nodes at a few key moments (i.e. opening and closing the model)
    conn = project.connect()

    sqls = [
        """DELETE from Nodes where is_centroid=0 and
                                      (SELECT count(*) FROM links WHERE a_node = node_id OR b_node = node_id) = 0;"""
    ]

    for sql in sqls:
        conn.execute(sql)
    conn.commit()
    conn.close()
