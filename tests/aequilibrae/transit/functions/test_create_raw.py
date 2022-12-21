import unittest

from aequilibrae.transit.functions.create_raw import create_raw_shapes


class TestCreateRaw(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_create_raw(self):
        create_raw_shapes(agency_id=1, select_patterns={})


if __name__ == "__name__":
    unittest.main()
