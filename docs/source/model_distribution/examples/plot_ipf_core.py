"""
.. _plot_ipf_core:

Running IPF with NumPy array
============================

In this example, we show how to use `aequilibrae.distribution.ipf_core`, an alternative to 
`aequilibrae.distribution.Ipf` for all those who don't want to run an IPF procedure without
creating a model or using data types such as Aequilibrae Matrix.

Let's consider that you have an OD-matrix, the future production and future attraction values.

The data used in this example comes from Table 5.6 in 
`Ortúzar & Willumsen (2011) <https://www.wiley.com/en-us/Modelling+Transport%2C+4th+Edition-p-9780470760390>`_.

"""

# %%
# .. admonition:: References
#
#   * :doc:`../IPF_benchmark`

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.distribution.ipf_core`

# %%

# Imports
import numpy as np

from aequilibrae.distribution.ipf_core import ipf_core

# sphinx_gallery_thumbnail_path = '../source/_images/ipf.png'

# %%
matrix = np.array([[5, 50, 100, 200], [50, 5, 100, 300], [50, 100, 5, 100], [100, 200, 250, 20]], dtype="float64")
future_prod = np.array([400, 460, 400, 702], dtype="float64")
future_attr = np.array([260, 400, 500, 802], dtype="float64")

# %%
# Given our use of default parameter values in the other application of IPF, we should set
# `tolerance` value to obtain the same result.
num_iter, gap = ipf_core(matrix, future_prod, future_attr, tolerance=0.0001)

# %%
# Let's print our updated matrix
matrix

# %%
# Notice that the results are the same as in :ref:`plot_ipf_without_model`, and this is no
# such coincidence. Under the hood, when we call `aequilibrae.distribution.Ipf`, we are actually
# calling the `ipf_core function`. For all those who only want to use AequilibraE's IPF procedure
# regardless of the purpose, this is the method for you!
