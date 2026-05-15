import os

import numpy as np
import pandas as pd
import pytest

from aequilibrae.distribution import Ipf


def test_fit(sioux_falls_test):
    mats = sioux_falls_test.matrices
    mats.update_database()
    seed = mats.get_matrix("SiouxFalls_omx")
    seed.computational_view("matrix")

    rows = np.random.rand(seed.zones)[:] * 1000
    cols = np.random.rand(seed.zones)[:] * 1000
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

    assert np.isclose(np.nansum(result.matrix_view), np.nansum(vectors["rows"]), rtol=0.0001), "Ipf did not converge"
    assert fratar.parameters["convergence level"] > fratar.gap, "Ipf did not converge"

    mr = fratar.save_to_project("my_matrix_ipf", "my_matrix_ipf.aem")

    assert os.path.isfile(os.path.join(mats.fldr, "my_matrix_ipf.aem")), "Did not save file to the appropriate place"
    assert mr.procedure_id == fratar.procedure_id, "procedure ID saved wrong"
