from unittest import TestCase
from tempfile import gettempdir
from os.path import join
from uuid import uuid4
from aequilibrae.utils.create_example import create_example
from aequilibrae.utils.create_delaunay_network import create_delaunay_network


class Test(TestCase):
    def test_create_delaunay_network(self):
        proj = create_example(join(gettempdir(), uuid4().hex))

        with self.assertRaises(ValueError):
            create_delaunay_network('nodes')

        create_delaunay_network()

        self.assertEqual(59, proj.conn.execute('select count(*) from delaunay_network').fetchone()[0])
        create_delaunay_network('network', True)

        self.assertEqual(62, proj.conn.execute('select count(*) from delaunay_network').fetchone()[0])
        with self.assertRaises(ValueError):
            create_delaunay_network()
