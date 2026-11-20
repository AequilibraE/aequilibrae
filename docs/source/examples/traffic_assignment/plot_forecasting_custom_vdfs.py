"""
.. _example_usage_forecasting_custom_vdfs:

Forecasting with a custom VDF
=============================

In this example, we use the built-in BPR volume-delay function (VDF) for the
base-year assignment in Sioux Falls. We then estimate future demand and use a
custom quadratic VDF for the future-year assignment with select link analysis.

See the :ref:`original forecasting example <example_usage_forecasting>` for the
full model setup, including gravity model calibration and trip length frequency
plots.
"""

from dataclasses import replace
from os.path import join
from tempfile import gettempdir
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aequilibrae.distribution import Ipf
from aequilibrae.examples import create_example
from aequilibrae.traffic import VDF, TrafficAssignment, TrafficClass
from aequilibrae.traffic.vdf import bpr
from aequilibrae.utils import basic_config

# %%
# Base-year assignment with the built-in BPR VDF
# ----------------------------------------------
# We use the built-in BPR VDF to run the base-year assignment with skims for
# ``free_flow_time`` and ``distance``.

# We create the example project inside our temp folder.
fldr = join(gettempdir(), uuid4().hex)
project = create_example(fldr)

# We'll also apply a basic logging configuration.
basic_config()

network = project.road.build_network()
proj_matrices = project.matrices

# Load the demand matrix core for computation.
demand = proj_matrices.get_matrix("demand.omx", subset=["matrix"])

# Create the assignment class
assigclass = TrafficClass(
    name="car",
    mode="c",
    cost="free_flow_time",
    demand=demand,
    skimming=["free_flow_time", "distance"],
    # Every node in Sioux Falls is a centroid, so paths must pass through centroids.
    graph_options={"block_centroid_flows": False},
)

# Configure the VDF, network fields, and assignment algorithm.
assig = TrafficAssignment(
    network=network,
    project=project,
    classes=[assigclass],
    vdf=bpr,
    name_mapping={"alpha": "b", "beta": "power"},
    time_field="free_flow_time",
    capacity_field="capacity",
    algorithm="bfw",
)
assig.execute(max_iterations=1000, rgap=0.001)

# Save the results to the project database and the skims to OMX.
assig.save_results("base_year_assignment")
assig.save_skims("base_year_assignment_skims", which_ones="all", format="omx")

# %%
# Future-year demand with IPF
# ---------------------------
# We apply random growth factors to the base-year origins and destinations, then
# balance their totals. We use iterative proportional fitting (IPF) to fit the base-year
# demand matrix to these future-year vectors.

origins = np.sum(demand["matrix"], axis=1)
destinations = np.sum(demand["matrix"], axis=0)

# Then grow them by up to 10% and balance the totals.
orig = origins * (1 + np.random.rand(origins.shape[0]) / 10)
dest = destinations * (1 + np.random.rand(origins.shape[0]) / 10)
dest *= orig.sum() / dest.sum()

vectors = pd.DataFrame({"origins": orig, "destinations": dest}, index=demand.index[:])
ipf = Ipf(
    project=project,
    matrix=demand,
    vectors=vectors,
    column_field="destinations",
    row_field="origins",
    nan_as_zero=True,
)
ipf.fit()

# Save the balanced matrix into the project's matrix folder.
ipf.save_to_project(file_name="demand_ipfd.omx")

# %%
# Define a custom quadratic VDF
# -----------------------------
# The VDF and its derivative write directly to their output arrays, ``congested_time``
# and ``delta``. See :ref:`creating_custom_vdfs` for the callable signatures and other
# ways to define a custom VDF.


def quadratic(congested_time, link_flows, fftime, capacity, cores, a, b):
    congested_time[:] = fftime * (a * (link_flows / capacity) ** 2 + b * (link_flows / capacity) + 1)


def quadratic_derivative(delta, link_flows, fftime, capacity, cores, a, b):
    delta[:] = fftime * (2 * a * (link_flows / capacity) + b) / capacity


vdf = VDF(
    "quadratic",
    quadratic,
    {"a": {"fill_NA": 0.15, "bounds": (0, float("inf"))}, "b": {"fill_NA": 0.1}},
    quadratic_derivative,
)

# %%
# Future-year assignment with the custom VDF
# ------------------------------------------
# We use the demand matrix from IPF and select two sets of links for select link analysis.
future_demand = proj_matrices.get_matrix("demand_ipfd.omx", subset=["matrix"])
select_links = {
    "Leaving node 1": [(1, 1), (2, 1)],
    "Random nodes": [(3, 1), (5, 1)],
}

# %%
# .. note::
#
#    As we are executing the select link analysis on a particular ``TrafficClass``, we should set the
#    links we want to analyse. The input is a dictionary with strings as keys and lists of tuples
#    as values, so that each entry represents a separate set of selected links to compute.
#
#    ``select_link_dict = {"set_name": [(link_id1, direction1), ..., (link_id, direction)]}``
#
#    The string name will name the set of links, and the list of tuples is the list of selected links
#    in the form ``(link_id, direction)``, as it occurs in the :ref:`Graph <aequilibrae-graphs>`.
#
#    Direction can be one of 0, 1, or -1, where 0 denotes bi-directionality.

# %%
# We reuse the base-year traffic class settings with the future-year demand and select link sets.
# Unlike the base-year BPR assignment, which maps its parameters to network fields,
# this assignment uses the same values of ``a`` and ``b`` on every link.
assigclass = replace(assigclass, demand=future_demand, select_link_sets=select_links)

# Configure the VDF, network fields, and assignment algorithm.
assig = TrafficAssignment(
    network=network,
    project=project,
    classes=[assigclass],
    vdf=vdf,
    name_mapping={"a": 0.15, "b": 1.0},
    time_field="free_flow_time",
    capacity_field="capacity",
    algorithm="bfw",
)
assig.execute(max_iterations=500, rgap=0.00001)

# %%
# To save our select link results, all we need to do is provide it with a name.
# In addition to exporting the select link flows, it also exports the Select Link matrices in OMX format.
assig.save_select_link_results("select_link_analysis")

# %%
# .. note::
#
#    Say we just want to save our select link flows, we can call: ``assig.save_select_link_flows("just_flows")``
#
#    Or if we just want the select link matrices: ``assig.save_select_link_matrices("just_matrices")``
#
#    Internally, the ``save_select_link_results`` calls both of these methods at once.

# %%
# We can export the results to CSV or AequilibraE Data, but let's put it directly into the results database
assig.save_results("future_year_assignment")

# %%
# And save the skims
assig.save_skims("future_year_assignment_skims", which_ones="all", format="omx")

# %%
# Run convergence study
# ~~~~~~~~~~~~~~~~~~~~~

df = assig.report()
x = df.iteration.values
y = df.rgap.values

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(x, y, "k--")
plt.yscale("log")
plt.grid(True, which="both")
plt.xlabel("Iterations")
plt.ylabel("Relative Gap")
plt.show()

# %%
# Close the project
project.close()

# %%
