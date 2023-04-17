"""
.. _plot_ipf_with_empty_model:

Running IPF with empty AequilibraE model
========================================

In this example, we show you how to use AquilibraE's IPF function with an empty model.
his is a compliment to the application in :ref:`Trip Distribution <example_usage_distribution>`.

Let's consider that you have an OD-matrix, the future production and future attraction values.
*How would your trip distribution matrix using IPF look like?*
The data used in this example comes from Table 5.6 in [ORW2011]_.
"""

# %%
# Imports
import numpy as np

from aequilibrae.distribution import Ipf
from aequilibrae.project import Project
from os.path import join
from tempfile import gettempdir
from uuid import uuid4
from aequilibrae.matrix import AequilibraeMatrix, AequilibraeData

# %%
folder = join(gettempdir(), uuid4().hex)
project = Project()
project.new(folder)

# %%
matrix = np.array([[5, 50, 100, 200], [50, 5, 100, 300], [50, 100, 5, 100], [100, 200, 250, 20]], dtype="float64")
future_prod = np.array([400, 460, 400, 702], dtype="float64")
future_attr = np.array([260, 400, 500, 802], dtype="float64")

num_zones = matrix.shape[0]

# %%
mtx = AequilibraeMatrix()
mtx.create_empty(file_name=join(folder, "matrix.aem"), zones=num_zones)
mtx.index[:] = np.arange(1, num_zones + 1)[:]
mtx.matrices[:, :, 0] = matrix[:]
mtx.computational_view()

# %%
args = {
    "entries": mtx.index.shape[0],
    "field_names": ["productions", "attractions"],
    "data_types": [np.float64, np.float64],
    "file_path": join(folder, "vectors.aem"),
}

vectors = AequilibraeData()
vectors.create_empty(**args)

vectors.productions[:] = future_prod[:]
vectors.attractions[:] = future_attr[:]

vectors.index[:] = mtx.index[:]

# %%
args = {
    "matrix": mtx,
    "rows": vectors,
    "row_field": "productions",
    "columns": vectors,
    "column_field": "attractions",
    "nan_as_zero": True,
}
fratar = Ipf(**args)
fratar.fit()

# %%
fratar.output.matrix_view

# %%
for line in fratar.report:
    print(line)
