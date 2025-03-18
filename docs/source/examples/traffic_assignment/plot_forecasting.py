"""
.. _example_usage_forecasting:

Forecasting
===========

In this example, we present a full forecasting workflow for the Sioux Falls example model.

We start creating the skim matrices, running the assignment for the base-year, and then
distributing these trips into the network. Later, we estimate a set of future demand vectors
which are going to be the input of a future year assignnment with select link analysis.
"""
# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.paths.Graph`
#     * :func:`aequilibrae.paths.TrafficClass`
#     * :func:`aequilibrae.paths.TrafficAssignment`
#     * :func:`aequilibrae.distribution.Ipf`
#     * :func:`aequilibrae.distribution.GravityCalibration`
#     * :func:`aequilibrae.distribution.GravityApplication`
#     * :func:`aequilibrae.distribution.SyntheticGravityModel`

# %%

# Imports
from uuid import uuid4
from os.path import join
from tempfile import gettempdir

import pandas as pd

from aequilibrae.utils.create_example import create_example
# sphinx_gallery_thumbnail_number = 3
# %%

# We create the example project inside our temp folder
fldr = join(gettempdir(), uuid4().hex)

project = create_example(fldr)
logger = project.logger

# %%
# Traffic assignment with skimming
# --------------------------------
# In this step, we'll set the skims for the variable ``free_flow_time``, and execute the
# traffic assignment for the base-year.

from aequilibrae.paths import TrafficAssignment, TrafficClass

# %%

# We build all graphs
project.network.build_graphs()
# We get warnings that several fields in the project are filled with NaNs. 
# This is true, but we won't use those fields.

# We grab the graph for cars
graph = project.network.graphs["c"]

# Let's say we want to minimize the free_flow_time
graph.set_graph("free_flow_time")

# And will skim time and distance while we are at it
graph.set_skimming(["free_flow_time", "distance"])

# And we will allow paths to be computed going through other centroids/centroid connectors
# required for the Sioux Falls network, as all nodes are centroids
graph.set_blocked_centroid_flows(False)

# %%
# Let's get the demand matrix directly from the project record, and inspect what matrices we have in the project.
proj_matrices = project.matrices
proj_matrices.list()

# %%
# We get the demand matrix, and prepare it for computation
demand = proj_matrices.get_matrix("demand_omx")
demand.computational_view(["matrix"])

# %%
# Let's perform the traffic assignment

# Create the assignment class
assigclass = TrafficClass(name="car", graph=graph, matrix=demand)

assig = TrafficAssignment()

# We start by adding the list of traffic classes to be assigned
assig.add_class(assigclass)

# Then we set these parameters, which an only be configured after adding one class to the assignment
assig.set_vdf("BPR")  # This is not case-sensitive 

# Then we set the volume delay function and its parameters
assig.set_vdf_parameters({"alpha": "b", "beta": "power"})

# The capacity and free flow travel times as they exist in the graph
assig.set_capacity_field("capacity")
assig.set_time_field("free_flow_time")

# And the algorithm we want to use to assign
assig.set_algorithm("bfw")

# Since we haven't checked the parameters file, let's make sure convergence criteria is good
assig.max_iter = 1000
assig.rgap_target = 0.001

# we then execute the assignment
assig.execute()

# %%
# After finishing the assignment, we can easily see the convergence report.
convergence_report = assig.report()
convergence_report.head()

# %%
# And we can also see the results of the assignment
results = assig.results()
results.head()

# %%
# We can export our results to CSV or get a Pandas DataFrame, but let's put it directly into the results database
assig.save_results("base_year_assignment")

# %%
# And save the skims
assig.save_skims("base_year_assignment_skims", which_ones="all", format="omx")

# %%
# Trip distribution
# -----------------
# First, let's have a function to plot the Trip Length Frequency Distribution.
from math import log10, floor
import matplotlib.pyplot as plt

# %%
def plot_tlfd(demand, skim, name):
    plt.clf()
    b = floor(log10(skim.shape[0]) * 10)
    n, bins, patches = plt.hist(
        np.nan_to_num(skim.flatten(), 0),
        bins=b,
        weights=np.nan_to_num(demand.flatten()),
        density=False,
        facecolor="g",
        alpha=0.75,
    )

    plt.xlabel("Trip length")
    plt.ylabel("Probability")
    plt.title(f"Trip-length frequency distribution for {name}")
    return plt

# %%
# Calibration
# ~~~~~~~~~~~
# We will calibrate synthetic gravity models using the skims for ``free_flow_time`` that we just generated

import numpy as np
from aequilibrae.distribution import GravityCalibration

# %%
# We need the demand matrix and to prepare it for computation
demand = proj_matrices.get_matrix("demand_aem")
demand.computational_view(["matrix"])

# %%
# We also need the skims we just saved into our project
imped = proj_matrices.get_matrix("base_year_assignment_skims_car")

# We can check which matrix cores were created for our skims to decide which one to use
imped.names

# %%
# Where ``free_flow_time_final`` is actually the congested time for the last iteration
#
# But before using the data, let's get some impedance for the intrazonals.
# Let's assume it is 75% of the closest zone.
imped_core = "free_flow_time_final"
imped.computational_view([imped_core])

# If we run the code below more than once, we will be overwriting the diagonal values with non-sensical data
# so let's zero it first
np.fill_diagonal(imped.matrix_view, 0)

# We compute it with a little bit of NumPy magic
intrazonals = np.amin(imped.matrix_view, where=imped.matrix_view > 0, initial=imped.matrix_view.max(), axis=1)
intrazonals *= 0.75

# Then we fill in the impedance matrix
np.fill_diagonal(imped.matrix_view, intrazonals)

# %%
# Since we are working with an OMX file, we cannot overwrite a matrix on disk. 
# So let's give it a new name to save.
imped.save(names=["final_time_with_intrazonals"])

# %%
# This also updates these new matrices as those being used for computation
imped.view_names

# %%
# Let's calibrate our Gravity Model
for function in ["power", "expo"]:
    gc = GravityCalibration(matrix=demand, impedance=imped, function=function, nan_as_zero=True)
    gc.calibrate()
    model = gc.model
    # We save the model
    model.save(join(fldr, f"{function}_model.mod"))

    _ = plot_tlfd(gc.result_matrix.matrix_view, imped.matrix_view, f"{function} model")

    # We can save the result of applying the model as well
    # We can also save the calibration report
    with open(join(fldr, f"{function}_convergence.log"), "w") as otp:
        for r in gc.report:
            otp.write(r + "\n")

# %%
# And let's plot a trip length frequency distribution for the demand itself
plt = plot_tlfd(demand.matrix_view, imped.matrix_view, "demand")
plt.show()

# %%
# Forecast
# --------
# We create a set of 'future' vectors using some random growth factors.
# We apply the model for inverse power, as the trip frequency length distribution
# (TFLD) seems to be a better fit for the actual one.

# %%
from aequilibrae.distribution import Ipf, GravityApplication, SyntheticGravityModel

# %%
# Compute future vectors
# ~~~~~~~~~~~~~~~~~~~~~~
# 
# First thing to do is to compute the future vectors from our matrix.
origins = np.sum(demand.matrix_view, axis=1)
destinations = np.sum(demand.matrix_view, axis=0)

# Then grow them with some random growth between 0 and 10%, and balance them
orig = origins * (1 + np.random.rand(origins.shape[0]) / 10)
dest = destinations * (1 + np.random.rand(origins.shape[0]) / 10)
dest *= orig.sum() / dest.sum()

vectors = pd.DataFrame({"origins":orig, "destinations":dest}, index=demand.index[:])
# %%
# IPF for the future vectors
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let's balance the future vectors. The output of this step is going to be used later
# in the traffic assignment for future year.

args = {
    "matrix": demand,
    "vectors": vectors,
    "column_field": "destinations",
    "row_field": "origins",
    "nan_as_zero": True,
}

ipf = Ipf(**args)
ipf.fit()

# %%
# When saving our vector into the project, we'll get an output that it was recored
ipf.save_to_project(name="demand_ipfd", file_name="demand_ipfd.aem")
ipf.save_to_project(name="demand_ipfd_omx", file_name="demand_ipfd.omx")

# %%
# Impedance
# ~~~~~~~~~
# 
# Let's get the base-year assignment skim for car we created before and prepare it for computation
imped = proj_matrices.get_matrix("base_year_assignment_skims_car")
imped.computational_view(["final_time_with_intrazonals"])

# %%
# If we wanted the main diagonal to not be considered...

# np.fill_diagonal(imped.matrix_view, np.nan)

# %%
# Now we apply the Synthetic Gravity model
for function in ["power", "expo"]:
    model = SyntheticGravityModel()
    model.load(join(fldr, f"{function}_model.mod"))

    outmatrix = join(proj_matrices.fldr, f"demand_{function}_model.aem")
    args = {
        "impedance": imped,
        "vectors": vectors,
        "row_field": "origins",
        "model": model,
        "column_field": "destinations",
        "nan_as_zero": True,
    }

    gravity = GravityApplication(**args)
    gravity.apply()

    # We get the output matrix and save it to OMX too,
    gravity.save_to_project(name=f"demand_{function}_modeled", file_name=f"demand_{function}_modeled.omx")

# %%
# We update the matrices table/records and verify that the new matrices are indeed there
proj_matrices.update_database()
proj_matrices.list()

# %%
# Traffic assignment with Select Link Analysis
# --------------------------------------------
# We'll perform traffic assignment for the future year.
logger.info("\n\n\n TRAFFIC ASSIGNMENT FOR FUTURE YEAR WITH SELECT LINK ANALYSIS")

# %%
# Let's get our future demand matrix, which corresponds to the IPF result we just saved,
# and see what is the core we ended up getting. It should be ``matrix``.
demand = proj_matrices.get_matrix("demand_ipfd")
demand.names

# %%
# Let's prepare our data for computation
demand.computational_view("matrix")

# %%
# The future year assignment is quite similar to the one we did for the base-year.

# So, let's create the assignment class
assigclass = TrafficClass(name="car", graph=graph, matrix=demand)

assig = TrafficAssignment()

# Add at a list of traffic classes to be assigned
assig.add_class(assigclass)

assig.set_vdf("BPR")

# Set the volume delay function and its parameters
assig.set_vdf_parameters({"alpha": "b", "beta": "power"})

# Set the capacity and free flow travel times as they exist in the graph
assig.set_capacity_field("capacity")
assig.set_time_field("free_flow_time")

# And the algorithm we want to use to assign
assig.set_algorithm("bfw")

# Once again we haven't checked the parameters file, so let's make sure convergence criteria is good
assig.max_iter = 500
assig.rgap_target = 0.00001

# %%
# Now we select two sets of links to execute select link analysis.
select_links = {
    "Leaving node 1": [(1, 1), (2, 1)],
    "Random nodes": [(3, 1), (5, 1)],
}

# %%
# .. note::
# 
#    As we are executing the select link analysis on a particular ``TrafficClass``, we should set the
#    links we want to analyze. The input is a dictionary with string as keys and a list of tuples as
#    values, so that each entry represents a separate set of selected links to compute. 
#    
#    ``select_link_dict = {"set_name": [(link_id1, direction1), ..., (link_id, direction)]}``
#
#    The string name will name the set of links, and the list of tuples is the list of selected links
#    in the form ``(link_id, direction)``, as it occurs in the :ref:`Graph <aequilibrae-graphs>`.
#
#    Direction can be one of 0, 1, or -1, where 0 denotes bi-directionality.

# %%

# We call this command on the class we are analyzing with our dictionary of values
assigclass.set_select_links(select_links)

# we then execute the assignment
assig.execute()

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
