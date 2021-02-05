from os.path import join
from tempfile import gettempdir
from uuid import uuid4
from unittest import TestCase
from aequilibrae.utils.create_example import create_example
from aequilibrae import Parameters


class Test(TestCase):
    def test_create_example(self):
        par = Parameters()._default
        for p in ["nauru", "sioux_falls"]:
            pth = join(gettempdir(), f"proj_example_{p}_{uuid4().hex}")
            proj = create_example(pth, p)
            parproj = proj.parameters
            self.assertEqual(parproj.keys(), par.keys(), f"Wrong parameter keys for {p} example")
            proj.close()
