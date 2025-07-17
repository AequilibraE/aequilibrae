import unittest
import tempfile
import pathlib

from aequilibrae.utils.create_example import create_example


class TestScenarios(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(delete=False)
        self.root = pathlib.Path(self.tmp.name)
        self.sioux_falls = create_example(self.root / "sioux_falls", "sioux_falls")
        self.nauru = create_example(self.root / "sioux_falls" / "nauru", "nauru")
        self.coquimbo = create_example(self.root / "sioux_falls" / "coquimbo", "coquimbo")

        with self.sioux_falls.db_connection as conn:
            conn.executemany("INSERT INTO scenarios (scenario_name) VALUES (?)", [("nauru",), ("coquimbo",)])

        with self.nauru.db_connection as conn:
            conn.execute("DROP TABLE scenarios")

        with self.coquimbo.db_connection as conn:
            conn.execute("DROP TABLE scenarios")

    def tearDown(self):
        self.tmp.cleanup()

    def test_something(self):
        assert False
