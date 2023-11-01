import os
import tempfile
from unittest import TestCase

import numpy as np

from aequilibrae.matrix import AequilibraeData


class TestAequilibraEData(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_path = AequilibraeData().random_name()
        cls.args = {
            "file_path": cls.file_path,
            "entries": 100,
            "field_names": ["d", "data2", "data3"],
            "data_types": [np.float64, np.float32, np.int8],
        }

    def test___init__(self):
        # Generates the dataset
        dataset = AequilibraeData()
        dataset.create_empty(**self.args)

        dataset.index[:] = np.arange(dataset.entries) + 100
        dataset.d[:] = dataset.index[:] ** 2
        if dataset.index[70] != 170:
            self.fail()

        if int(dataset.d[70]) != 28900:
            self.fail()

        # removes the dataset
        del dataset

    def test_load(self):
        # re-imports the dataset
        self.ad = AequilibraeData()
        self.ad.load(self.file_path)

        # checks if the values were properly saved
        if self.ad.index[70] != 170:
            self.fail("Value for data index test was not as expected")

        if int(self.ad.d[70]) != 28900:
            self.fail("Value for data field test was not as expected")

        for f in self.ad.fields:
            if f not in self.args["field_names"]:
                self.fail("Could not retrieve all fields")

    def test_export(self):
        self.test_load()
        temp_name = os.path.join(tempfile.gettempdir(), "aequilibrae_data_example.csv")
        self.ad.export(temp_name)
