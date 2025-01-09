"""
.. _plot_ipf_core:

Running IPF with NumPy array
============================

In this example, we show how to use ``aequilibrae.distribution.ipf_core``, a high-performance 
alternative for all those who want to (re)balance values within a matrix making direct use of
growth factors. ``ipf_core`` was built to suit countless applications rather than being limited
to trip distribution.

We demonstrate the usage of ``ipf_core`` with a 4x4 matrix with 64-bit data, which is indeed very
small. Additionally, a more comprehensive discussion of the algorithm's performance
with a 32-bit or 64-bit seed matrices is provided in :doc:`../IPF_benchmark`.

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
# Notice that the matrix value was updated, and results are the same as in :ref:`plot_ipf_without_model`
# - and this is no coincidence. Under the hood, when we call ``aequilibrae.distribution.Ipf``, we 
# are actually calling the ``ipf_core`` method.
