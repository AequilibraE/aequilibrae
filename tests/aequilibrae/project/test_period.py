from unittest import TestCase
from sqlite3 import IntegrityError
import os
from shutil import copytree, rmtree
from random import randint, random
import uuid
from tempfile import gettempdir
from shapely.geometry import Point
import shapely.wkb
from aequilibrae.project import Project
import pandas as pd

from ...data import siouxfalls_project


class TestPeriod(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

        self.proj_dir = os.path.join(gettempdir(), uuid.uuid4().hex)
        copytree(siouxfalls_project, self.proj_dir)

        self.project = Project()
        self.project.open(self.proj_dir)
        self.network = self.project.network

        for num in range(2, 6):
            self.project.network.periods.new_period(num, num, num, "test")

    def tearDown(self) -> None:
        self.project.close()
        try:
            rmtree(self.proj_dir)
        except Exception as e:
            print(f"Failed to remove at {e.args}")

    def test_save_and_assignment(self):
        periods = self.network.periods
        nd = randint(2, 5)
        period = periods.get(nd)

        with self.assertRaises(AttributeError):
            period.modes = "abc"

        with self.assertRaises(AttributeError):
            period.link_types = "default"

        with self.assertRaises(AttributeError):
            period.period_id = 2

        period.period_description = "test"
        self.assertEqual(period.period_description, "test")

        period.save()

        expected = pd.DataFrame(
            {
                "period_id": [1, nd],
                "period_start": [0, nd],
                "period_end": [86400, nd],
            }
        )
        expected["period_description"] = "test"
        expected.at[0, "period_description"] = "Default time period, whole day"

        pd.testing.assert_frame_equal(periods.data, expected)

    def test_data_fields(self):
        periods = self.network.periods

        period = periods.get(1)

        fields = sorted(period.data_fields())
        dt = self.project.conn.execute("pragma table_info(periods)").fetchall()

        actual_fields = sorted([x[1] for x in dt if x[1] != "ogc_fid"])

        self.assertEqual(fields, actual_fields, "Period has unexpected set of fields")

    def test_renumber(self):
        periods = self.network.periods

        period = periods.get(1)

        with self.assertRaises(ValueError):
            period.renumber(1)

        num = randint(25, 2000)
        with self.assertRaises(ValueError):
            period.renumber(num)

        new_period = periods.new_period(num, 0, 0, "test")
        new_period.renumber(num + 1)

        self.assertEqual(new_period.period_id, num + 1)
