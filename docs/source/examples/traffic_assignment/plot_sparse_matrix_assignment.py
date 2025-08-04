"""
.. _example_assign_sparse:

Assigning sparse matrices
=========================

Modern Activity-Based models (and even some trip-based and tour-based ones) result on incredibly sparse
demand matrices, which opens up a significant opportunity to save time during assignment by using early-exiting
during the path-computation phase of assignment.

To take advantage of this, while still computing assignment skims, AequilibraE has a built-in method to
skim the last iteration after the assignment is done.
"""

# %%
# .. admonition:: Technical references
#
#   * :doc:`../../traffic_assignment/assignment_procedures`

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.paths.Graph`
#     * :func:`aequilibrae.paths.TrafficClass`
#     * :func:`aequilibrae.paths.TrafficAssignment`

# %%

# Imports
from os.path import join
from tempfile import gettempdir
from uuid import uuid4

from aequilibrae.utils.create_example import create_example
from aequilibrae.paths import TrafficAssignment, TrafficClass

# sphinx_gallery_thumbnail_path = '../source/_images/sparse_matrix.png'

# %%

# We create the example project inside our temp folder
fldr = join(gettempdir(), uuid4().hex)

project = create_example(fldr)
logger = project.logger

# %%
# Traffic assignment
# ------------------

# %%
# We build all graphs
project.network.build_graphs()
# We get warnings that several fields in the project are filled with NaNs. 
# This is true, but we won't use those fields.

# We grab the graph for cars
graph = project.network.graphs["c"]

# Let's say we want to minimize the free_flow_time
graph.set_graph("free_flow_time")

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

# Let's set parameters that make this example run very fast
assig.max_iter = 10
assig.rgap_target = 0.01

# we then execute the assignment
assig.execute()

# %%
# After finishing the assignment, we can skim the last iteration
skims = assig.skim_congested(["distance"], return_matrices=True)

# Skims are returned as a dictionary, with the class names as keys
# Let's see all skims we have inside it:
print(skims["car"].names)

# %%
# We can save the skims, but we need to choose to only save the final ones, as the blended were not generated
assig.save_skims("base_year_assignment_skims", which_ones="final", format="omx")

# %%
# Close the project
project.close()
