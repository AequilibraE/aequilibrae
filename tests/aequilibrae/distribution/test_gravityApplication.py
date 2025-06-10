from unittest import TestCase

import numpy as np
import pandas as pd

from aequilibrae.distribution import SyntheticGravityModel, GravityApplication
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.parameters import Parameters

zones = 10


idx = np.arange(zones) + 100
rows = idx + np.random.rand(zones)[:]
cols = idx + np.random.rand(zones)[:]

# balance vectors
cols *= np.sum(rows) / np.sum(cols)

vectors = pd.DataFrame({"rows": rows, "columns": cols}, index=idx)  # row vector
args = {"entries": zones, "field_names": ["rows"], "data_types": [np.float64], "memory_mode": True}


matrix = AequilibraeMatrix()
matrix.create_empty(zones=zones, matrix_names=["impedance"], memory_only=True)

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


def test_gravity_application():
    proj = Parameters()
    args = {
        "impedance": matrix,
        "vectors": vectors,
        "row_field": "rows",
        "column_field": "columns",
    }

    models = [("EXPO", model_expo), ("POWER", model_power), ("GAMMA", model_gamma)]

    for model_name, model_obj in models:
        args["model"] = model_obj
        distributed_matrix = GravityApplication(project=proj, **args)
        distributed_matrix.apply()

        if distributed_matrix.gap > distributed_matrix.parameters["convergence level"]:
            raise ValueError(f"Gravity application did not converge for model {model_name}")
