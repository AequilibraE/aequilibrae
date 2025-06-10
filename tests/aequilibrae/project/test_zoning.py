from os.path import join
from tempfile import gettempdir
from unittest import TestCase
from uuid import uuid4

from shapely.geometry import Polygon

from aequilibrae.utils.create_example import create_example


class TestZoning(TestCase):
    def setUp(self) -> None:
        self.temp_proj_folder = join(gettempdir(), uuid4().hex)
        self.proj = create_example(self.temp_proj_folder, "coquimbo")
        with self.proj.db_connection as conn:
            conn.execute("DELETE FROM links WHERE name LIKE 'centroid connector%'")
            conn.execute("DELETE FROM nodes WHERE is_centroid=1;")
            conn.commit()
            self.centroids = conn.execute("SELECT COUNT(node_id) FROM nodes WHERE is_centroid=1;").fetchone()[0]

        self.zoning = self.proj.zoning

    def tearDown(self) -> None:
        self.proj.close()

    def test_add_centroid(self):
        self.zoning.add_centroids()

        with self.proj.db_connection as conn:
            num_centroids = conn.execute("SELECT COUNT(node_id) FROM nodes WHERE is_centroid=1;").fetchone()[0]

        self.assertGreater(num_centroids, self.centroids, "Centroids should've been added.")

    def test_connect_mode(self):
        links_before = self.proj.network.links.data.shape[0]

        self.zoning.add_centroids()

        self.zoning.connect_mode(mode_id="c", connectors=1)

        links_after = self.proj.network.links.data.shape[0]
        self.assertGreater(links_after, links_before, "Centroid connectors should've been added.")

    def test_coverage(self):
        cov = self.zoning.coverage()

        self.assertTrue(isinstance(cov, Polygon), "Coverage geometry type is incorrect")

    def test_create_zoning_layer(self):
        tables = [
            "zones",
            "idx_zones_geometry",
            "idx_zones_geometry_node",
            "idx_zones_geometry_parent",
            "idx_zones_geometry_rowid",
        ]
        with self.proj.db_connection as conn:
            for table in tables:
                conn.execute(f"DROP TABLE IF EXISTS {table};")
            conn.execute("DELETE FROM attributes_documentation WHERE name_table LIKE 'zones'")

            fields = [x[1] for x in conn.execute("PRAGMA table_info(zones);").fetchall()]

        self.assertEqual(fields, [], "Zone table fields still exist")

        zoning = self.proj.zoning
        zoning.create_zoning_layer()

        with self.proj.db_connection as conn:
            fields = [x[1] for x in conn.execute("PRAGMA table_info(zones);").fetchall()]

        self.assertGreater(len(fields), 0, "Zone table exists and has its fields.")
