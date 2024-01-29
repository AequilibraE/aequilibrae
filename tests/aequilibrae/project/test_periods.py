import os
import uuid
from copy import copy, deepcopy
from random import randint, random
from shutil import copytree, rmtree
from tempfile import gettempdir
from unittest import TestCase, TestLoader as _TestLoader
import pandas as pd

import shapely.wkb
from shapely.geometry import Point

from aequilibrae.project import Project
from ...data import siouxfalls_project

_TestLoader.sortTestMethodsUsing = None


class TestPeriods(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

        self.proj_dir = os.path.join(gettempdir(), uuid.uuid4().hex)
        copytree(siouxfalls_project, self.proj_dir)

        self.project = Project()
        self.project.open(self.proj_dir)
        self.network = self.project.network
        self.curr = self.project.conn.cursor()

    def tearDown(self) -> None:
        self.project.close()
        del self.curr
        try:
            rmtree(self.proj_dir)
        except Exception as e:
            print(f"Failed to remove at {e.args}")

    def test_get(self):
        periods = self.network.periods
        for num in range(2, 6):
            self.project.network.periods.new_period(num, num, num, "test")

        nd = randint(2, 5)
        period = periods.get(nd)

        self.assertEqual(period.period_id, nd, "get period returned wrong object")

        # Make sure that if we renumber itg we would not get it again
        period.renumber(200)
        with self.assertRaises(ValueError):
            period = periods.get(nd)

    def test_fields(self):
        periods = self.network.periods
        f_editor = periods.fields

        fields = sorted(f_editor.all_fields())
        self.curr.execute("pragma table_info(periods)")
        dt = self.curr.fetchall()

        actual_fields = sorted({x[1] for x in dt})
        self.assertEqual(fields, actual_fields, "Table editor is weird for table periods")

    def test_copy(self):
        periods = self.network.periods
        with self.assertRaises(Exception):
            _ = copy(periods)
        with self.assertRaises(Exception):
            _ = deepcopy(periods)

    def test_save(self):
        periods = self.network.periods
        for num in range(2, 6):
            self.project.network.periods.new_period(num, num, num, "test")

        periods.save()

        expected = pd.DataFrame(
            {
                "period_id": [1, 2, 3, 4, 5],
                "period_start": [0, 2, 3, 4, 5],
                "period_end": [86400, 2, 3, 4, 5],
            }
        )
        expected["period_description"] = "test"
        expected.at[0, "period_description"] = "Default time period, whole day"

        # breakpoint()
        pd.testing.assert_frame_equal(periods.data, expected)
