from unittest import TestCase
from tempfile import gettempdir
from os.path import join
from uuid import uuid4
from aequilibrae.utils.create_example import create_example
from aequilibrae.utils.create_delaunay_network import DelaunayAnalysis


class TestDelaunayAnalysis(TestCase):
    def setUp(self) -> None:
        self.proj = create_example(join(gettempdir(), uuid4().hex))

    def tearDown(self) -> None:
        self.proj.close()

    def test_create_delaunay_network(self):
        da = DelaunayAnalysis(self.proj)
        with self.assertRaises(ValueError):
            da.create_network("nodes")

        da.create_network()
        self.assertEqual(59, self.proj.conn.execute("select count(*) from delaunay_network").fetchone()[0])

        da.create_network("network", True)
        self.assertEqual(62, self.proj.conn.execute("select count(*) from delaunay_network").fetchone()[0])

        with self.assertRaises(ValueError):
            da.create_network()

    def test_assign_matrix(self):
        demand = self.proj.matrices.get_matrix("demand_omx")
        demand.computational_view(["matrix"])
        da = DelaunayAnalysis(self.proj)
        da.create_network()
        da.assign_matrix(demand, "delaunay_test")
