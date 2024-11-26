"""
.. _plot_ipf_without_model:

Running IPF without an AequilibraE model
========================================

In this example, we show you how to use AequilibraE's IPF function without a model.
This is a complement to the application in :ref:`example_usage_forecasting`.

Let's consider that you have an OD-matrix, the future production and future attraction values.

*How would your trip distribution matrix using IPF look like?*

The data used in this example comes from Table 5.6 in 
`Ortúzar & Willumsen (2011) <https://www.wiley.com/en-us/Modelling+Transport%2C+4th+Edition-p-9780470760390>`_.

"""
# %%
# .. admonition:: References
# 
#   * :ref:`all_about_aeq_matrices` 
#   * :ref:`validation`

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.matrix.AequilibraeMatrix`
#     * :func:`aequilibrae.distribution.Ipf`

# %%

# Imports
from os.path import join
from tempfile import gettempdir

import numpy as np
import pandas as pd

from aequilibrae.distribution import Ipf
from aequilibrae.matrix import AequilibraeMatrix
# sphinx_gallery_thumbnail_path = '../source/images/ipf.png'

# %%
folder = gettempdir()

# %%
matrix = np.array([[5, 50, 100, 200], [50, 5, 100, 300], 
                   [50, 100, 5, 100], [100, 200, 250, 20]], dtype="float64")
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

vectors = pd.DataFrame({"productions": future_prod, "attractions":future_attr}, index=mtx.index)
# %%
args = {
    "matrix": mtx,
    "vectors": vectors,
    "row_field": "productions",
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
    
