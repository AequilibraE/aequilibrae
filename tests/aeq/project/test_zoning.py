from shapely.geometry import Polygon


def zoning_setup(project):
    with project.db_connection as conn:
        conn.execute("DELETE FROM links WHERE name LIKE 'centroid connector%'")
        conn.execute("DELETE FROM nodes WHERE is_centroid=1;")
        conn.commit()
        centroids = conn.execute("SELECT COUNT(node_id) FROM nodes WHERE is_centroid=1;").fetchone()[0]

    return project, centroids


def test_add_centroid(coquimbo_example):
    proj, centroids = zoning_setup(coquimbo_example)
    proj.zoning.add_centroids()
    with proj.db_connection as conn:
        num_centroids = conn.execute("SELECT COUNT(node_id) FROM nodes WHERE is_centroid=1;").fetchone()[0]
    assert num_centroids > centroids, "Centroids should've been added."


def test_connect_mode(coquimbo_example):
    proj, _ = zoning_setup(coquimbo_example)
    links_before = proj.network.links.data.shape[0]
    proj.zoning.add_centroids()
    proj.zoning.connect_mode(mode_id="c", connectors=1)
    links_after = proj.network.links.data.shape[0]
    assert links_after > links_before, "Centroid connectors should've been added."


def test_coverage(coquimbo_example):
    proj, _ = zoning_setup(coquimbo_example)
    cov = proj.zoning.coverage()
    assert isinstance(cov, Polygon), "Coverage geometry type is incorrect"


def test_create_zoning_layer(coquimbo_example):
    proj, _ = zoning_setup(coquimbo_example)
    tables = [
        "zones",
        "idx_zones_geometry",
        "idx_zones_geometry_node",
        "idx_zones_geometry_parent",
        "idx_zones_geometry_rowid",
    ]
    with proj.db_connection_spatial as conn:
        for table in tables:
            conn.execute(f"DROP TABLE IF EXISTS {table};")
        conn.execute("DELETE FROM attributes_documentation WHERE name_table LIKE 'zones'")
        fields = [x[1] for x in conn.execute("PRAGMA table_info(zones);").fetchall()]
    assert fields == [], "Zone table fields still exist"
    zoning = proj.zoning
    zoning.create_zoning_layer()
    with proj.db_connection as conn:
        fields = [x[1] for x in conn.execute("PRAGMA table_info(zones);").fetchall()]
    assert len(fields) > 0, "Zone table exists and has its fields."
