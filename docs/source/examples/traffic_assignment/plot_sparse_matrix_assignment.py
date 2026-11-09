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
#     * :func:`aequilibrae.paths.graph`
#     * :func:`aequilibrae.paths.traffic_class.TrafficClass`
#     * :func:`aequilibrae.paths.traffic_assignment.TrafficAssignment`

# %%

# Imports
from os.path import join
from tempfile import gettempdir
from uuid import uuid4

from scipy.sparse import csr_array

from aequilibrae.matrix import from_dict
from aequilibrae.paths import TrafficAssignment, TrafficClass
from aequilibrae.paths.vdf import bpr
from aequilibrae.utils.create_example import create_example
from aequilibrae.utils.logging_utils import basic_config

# sphinx_gallery_thumbnail_path = '../source/_images/sparse_matrix.png'

# %%

# We create the example project inside our temp folder
fldr = join(gettempdir(), uuid4().hex)

project = create_example(fldr)

# %%

# We'll also apply a basic logging configuration.

basic_config()

# %%
# Traffic assignment
# ------------------

# %%
network = project.network

# %%
# Let's get the demand matrix directly from the project record, and inspect what matrices we have in the project.
proj_matrices = project.matrices
proj_matrices.get_available_matrices_and_metadata()

# %%
# We get the demand matrix, and prepare it for computation
demand = proj_matrices.get_matrix("demand.omx", subset=["matrix"])

# Copy the named matrices into sparse float64 storage. In practise these would
# be stored in the sparse OMX extension rather than as a dense matrix, then
# converted to a sparse format.
demand = from_dict(
    {name: csr_array(values, dtype="float64") for name, values in demand.items()},
    index=demand.index,
)

# %%
# Let's perform the traffic assignment

# Create the assignment class
assigclass = TrafficClass(
    name="car", mode="c", cost="free_flow_time", demand=demand, skimming=[], block_centroid_flows=False
)

# Configure the VDF, network fields, and assignment algorithm.
assig = TrafficAssignment(
    network=network,
    classes=[assigclass],
    vdf=bpr,
    name_mapping={"alpha": "b", "beta": "power"},
    time_field="free_flow_time",
    capacity_field="capacity",
    algorithm="bfw",
)
assig.execute(max_iterations=10, rgap=0.01)

# %%
# After finishing the assignment, we can skim the last iteration
skims = assig.skim_congested(["distance"], return_matrices=True)

# Skims are returned as a dictionary, with the class names as keys
# Let's see all skims we have inside it:
print(skims["car"].field_names)

# %%
# We can save the skims, but we need to choose to only save the final ones, as the blended were not generated
assig.save_skims("base_year_assignment_skims", which_ones="final", format="omx")

# %%
# Close the project
project.close()
