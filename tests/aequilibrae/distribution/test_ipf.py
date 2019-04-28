from unittest import TestCase

import numpy as np

from aequilibrae.distribution import Ipf
from aequilibrae.matrix import AequilibraEData
from aequilibrae.matrix import AequilibraeMatrix

zones = 100

# row vector
args = {"entries": zones, "field_names": ["rows"], "data_types": [np.float64], "memory_mode": True}
row_vector = AequilibraEData()
row_vector.create_empty(**args)
row_vector.rows[:] = np.random.rand(zones)[:] * 1000
row_vector.index[:] = np.arange(zones)[:]
# column vector
args["field_names"] = ["columns"]
column_vector = AequilibraEData()
column_vector.create_empty(**args)
column_vector.columns[:] = np.random.rand(zones)[:] * 1000
column_vector.index[:] = np.arange(zones)[:]
# balance vectors
column_vector.columns[:] = column_vector.columns[:] * (row_vector.rows.sum() / column_vector.columns.sum())

# seed matrix_procedures
name_test = AequilibraeMatrix().random_name()
args = {"file_name": name_test, "zones": zones, "matrix_names": ["seed"]}

matrix = AequilibraeMatrix()
matrix.create_empty(**args)
matrix.seed[:, :] = np.random.rand(zones, zones)[:, :]
matrix.computational_view(["seed"])
matrix.matrix_view[1, 1] = np.nan
matrix.index[:] = np.arange(zones)[:]


class TestIpf(TestCase):
    def test_fit(self):
        # The IPF per se
        args = {
            "matrix": matrix,
            "rows": row_vector,
            "row_field": "rows",
            "columns": column_vector,
            "column_field": "columns",
            "nan_as_zero": False,
        }

        fratar = Ipf(**args)
        fratar.fit()

        result = fratar.output
        if (np.nansum(result.matrix_view) - np.nansum(row_vector.data["rows"])) > 0.001:
            print(fratar.gap)
            for f in fratar.report:
                print(f)
            self.fail("Ipf did not converge")

        if fratar.gap > fratar.parameters["convergence level"]:
            print(fratar.gap)
            for f in fratar.report:
                print(f)
            self.fail("Ipf did not converge")
