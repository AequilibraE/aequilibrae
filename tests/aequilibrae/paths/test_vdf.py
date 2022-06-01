from unittest import TestCase
from aequilibrae.paths.vdf import VDF


class TestVDF(TestCase):
    def test_functions_available(self):
        v = VDF()
        self.assertEqual(
            v.functions_available(), ["bpr", "bpr2", "conical", "inrets"], "VDF class returning wrong availability"
        )
        self.assertEqual(v.apply_vdf, None, "VDF is missing term")
        self.assertEqual(v.apply_derivative, None, "VDF is missing term")

        with self.assertRaises(ValueError):
            v.function = "Cubic"

        with self.assertRaises(AttributeError):
            v.apply_vdf = isinstance
