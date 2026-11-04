from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aequilibrae.distribution import SyntheticGravityModel, GravityApplication
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.parameters import Parameters
from aequilibrae.project.data.matrices import Matrices


@pytest.fixture(scope="session")
def model_expo():
    model = SyntheticGravityModel()
    model.function = "EXPO"
    model.beta = 0.1
    return model


@pytest.fixture(scope="session")
def model_gamma():
    model = SyntheticGravityModel()
    model.function = "GAMMA"
    model.beta = 0.1
    model.alpha = -0.2
    return model


@pytest.fixture(scope="session")
def model_power():
    model = SyntheticGravityModel()
    model.function = "POWER"
    model.alpha = -0.2
    return model


def test_gravity_application(model_expo, model_gamma, model_power, tmp_path):
    zones = 10

    idx = np.arange(zones) + 100
    rows = idx + np.random.rand(zones)[:]
    cols = idx + np.random.rand(zones)[:]

    # balance vectors
    cols *= np.sum(rows) / np.sum(cols)

    vectors = pd.DataFrame({"rows": rows, "columns": cols}, index=idx)  # row vector

    matrix = AequilibraeMatrix(index=idx)
    matrix["impedance"] = np.random.rand(zones, zones)

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

        project = SimpleNamespace(matrices=Matrices(tmp_path))
        file_name = f"gravity_{model_name}.omx"
        distributed_matrix.save_to_project(f"gravity_{model_name}", file_name, project=project)
        metadata = project.matrices.get_available_matrices_and_metadata()[file_name]
        assert metadata["procedure"] == "Synthetic gravity trip distribution"
        assert metadata["procedure_id"] == distributed_matrix.procedure_id
        assert metadata["timestamp"] == distributed_matrix.procedure_date
        assert metadata["description"] == f"Synthetic gravity trip distribution. {model_name}"
        with project.matrices.get_matrix(file_name) as loaded:
            core = distributed_matrix.core_name
            np.testing.assert_array_equal(loaded[core], distributed_matrix.output[core])
