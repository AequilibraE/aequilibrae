import os
import tempfile
from unittest import TestCase

import numpy as np

from aequilibrae.distribution import SyntheticGravityModel, GravityApplication
from aequilibrae.matrix import AequilibraeData, AequilibraeMatrix
from aequilibrae.parameters import Parameters

zones = 10

# row vector
args = {"entries": zones, "field_names": ["rows"], "data_types": [np.float64], "memory_mode": True}

row_vector = AequilibraeData()
row_vector.create_empty(**args)
row_vector.index[:] = np.arange(row_vector.entries) + 100
row_vector.rows[:] = row_vector.index[:] + np.random.rand(zones)[:]

# column vector
args["field_names"] = ["columns"]
column_vector = AequilibraeData()
column_vector.create_empty(**args)
column_vector.index[:] = np.arange(column_vector.entries) + 100
column_vector.columns[:] = column_vector.index[:] + np.random.rand(zones)[:]

# balance vectors
column_vector.columns[:] = column_vector.columns[:] * (row_vector.rows.sum() / column_vector.columns.sum())

# Impedance matrix_procedures
name_test = os.path.join(tempfile.gettempdir(), "aequilibrae_matrix_test.aem")

args = {"file_name": name_test, "zones": zones, "matrix_names": ["impedance"]}

matrix = AequilibraeMatrix()
matrix.create_empty(**args)

# randoms = np.random.randint(5, size=(2, 4))
matrix.matrices[:, :, 0] = np.random.rand(zones, zones)[:, :]
matrix.index[:] = np.arange(matrix.zones) + 100
matrix.computational_view(["impedance"])

model_expo = SyntheticGravityModel()
model_expo.function = "EXPO"
model_expo.beta = 0.1

model_gamma = SyntheticGravityModel()
model_gamma.function = "GAMMA"
model_gamma.beta = 0.1
model_gamma.alpha = -0.2

model_power = SyntheticGravityModel()
model_power.function = "POWER"
model_power.alpha = -0.2


class TestGravityApplication(TestCase):
    def setUp(self):
        # GravityApplication requires an object that has a `parameters` attribute. `Parameters` fits
        # this requirement, so that we don't need to create a full project
        self.proj = Parameters()

    def test_apply(self):
        args = {
            "impedance": matrix,
            "rows": row_vector,
            "row_field": "rows",
            "columns": column_vector,
            "column_field": "columns",
        }

        models = [("EXPO", model_expo), ("POWER", model_power), ("GAMMA", model_gamma)]

        for model_name, model_obj in models:
            args["model"] = model_obj
            distributed_matrix = GravityApplication(project=self.proj, **args)
            distributed_matrix.apply()

            if distributed_matrix.gap > distributed_matrix.parameters["convergence level"]:
                self.fail(f"Gravity application did not converge for model {model_name}")
