import os

import numpy as np
import pandas as pd
import pytest

from aequilibrae.distribution import Ipf


def test_fit(sioux_falls_test):
    mats = sioux_falls_test.matrices
    seed = mats.get_matrix("SiouxFalls2.omx", subset=["matrix"])

    rows = np.random.rand(len(seed.index))[:] * 1000
    cols = np.random.rand(len(seed.index))[:] * 1000
    vectors = pd.DataFrame({"rows": rows, "columns": cols}, index=seed.index)

    vectors["columns"] *= vectors["rows"].sum() / vectors["columns"].sum()

    # The IPF per se
    args = {
        "matrix": seed,
        "vectors": vectors,
        "row_field": "rows",
        "column_field": "columns",
        "nan_as_zero": False,
    }

    with pytest.raises(TypeError):
        fratar = Ipf(data="test", test="data")
        fratar.fit()

    with pytest.raises(ValueError):
        fratar = Ipf(**args)
        fratar.parameters = ["test"]
        fratar.fit()

    fratar = Ipf(**args)
    fratar.fit()

    result = fratar.output

    assert np.isclose(np.nansum(result["matrix"]), np.nansum(vectors["rows"]), rtol=0.0001), "Ipf did not converge"
    assert fratar.parameters["convergence level"] > fratar.gap, "Ipf did not converge"

    fratar.save_to_project("my_matrix_ipf", "my_matrix_ipf.omx", project=sioux_falls_test)

    assert os.path.isfile(os.path.join(mats.folder, "my_matrix_ipf.omx")), "Did not save file to the appropriate place"
    metadata = mats.get_available_matrices_and_metadata()["my_matrix_ipf.omx"]
    assert metadata["procedure"] == "Iterative Proportional fitting"
    assert metadata["procedure_id"] == fratar.procedure_id
    assert metadata["timestamp"] == fratar.procedure_date
