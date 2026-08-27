import pytest
from shapely.geometry import Polygon


def zones_setup(project):
    with project.db_connection as conn:
        conn.execute("DELETE FROM links WHERE name LIKE 'centroid connector%'")
        conn.execute("DELETE FROM nodes WHERE is_centroid=1;")
        conn.commit()
        centroids = conn.execute("SELECT COUNT(node_id) FROM nodes WHERE is_centroid=1;").fetchone()[0]

    return project, centroids


def test_add_centroid(coquimbo_example):
    proj, centroids = zones_setup(coquimbo_example)
    proj.network.zones.add_centroids()
    with proj.db_connection as conn:
        num_centroids = conn.execute("SELECT COUNT(node_id) FROM nodes WHERE is_centroid=1;").fetchone()[0]
    assert num_centroids > centroids, "Centroids should've been added."


@pytest.mark.parametrize("bulk", [True, False])
def test_connect_mode(coquimbo_example, bulk):
    proj, _ = zones_setup(coquimbo_example)
    links_before = proj.network.links.data.shape[0]
    proj.network.zones.add_centroids()
    proj.network.zones.connect_mode(mode_id="c", connectors=1, bulk=bulk)
    links_after = proj.network.links.data.shape[0]
    assert links_after > links_before, "Centroid connectors should've been added."


def test_coverage_and_spatial_table_interfaces(coquimbo_example):
    proj, _ = zones_setup(coquimbo_example)
    zones = proj.network.zones
    cov = zones.coverage()
    assert isinstance(cov, Polygon), "Coverage geometry type is incorrect"
    assert isinstance(zones.extent(), Polygon)
    assert zones.has_zones
    assert len(zones) == len(zones.data)
    assert {zone.zone_id for zone in zones} == set(zones.data.zone_id)


def test_create_zones_table(coquimbo_example):
    proj, _ = zones_setup(coquimbo_example)
    tables = [
        "zones",
        "idx_zones_geometry",
        "idx_zones_geometry_node",
        "idx_zones_geometry_parent",
        "idx_zones_geometry_rowid",
    ]
    with proj.db_connection as conn:
        for table in tables:
            conn.execute(f"DROP TABLE IF EXISTS {table};")
        conn.execute("DELETE FROM attributes_documentation WHERE name_table LIKE 'zones'")
        fields = [x[1] for x in conn.execute("PRAGMA table_info(zones);").fetchall()]
    assert fields == [], "Zone table fields still exist"
    zones = proj.network.zones
    zones.create_zones_table()
    with proj.db_connection as conn:
        fields = [x[1] for x in conn.execute("PRAGMA table_info(zones);").fetchall()]
    assert len(fields) > 0, "Zone table exists and has its fields."
