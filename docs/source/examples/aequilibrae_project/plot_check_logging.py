"""
.. _useful-log-tips:

Checking AequilibraE's log
==========================

AequilibraE's log is a very useful tool to get more information about
what the software is doing under the hood.

Information such as Traffic Class and Traffic Assignment stats, and Traffic Assignment
outputs. If you have created your project's network from OSM, you will also find
information on the number of nodes, links, and the query performed to obtain the data.

In this example, we'll use Sioux Falls data to check the logs, but we strongly encourage
you to go ahead and download a place of your choice and perform a traffic assignment!
"""
# %%

# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae.utils.create_example import create_example
from aequilibrae.paths import TrafficAssignment, TrafficClass
from aequilibrae.paths.vdf import bpr
# sphinx_gallery_thumbnail_path = '../source/_images/logs_image.png'

# %%

# We create an empty project on an arbitrary folder
fldr = join(gettempdir(), uuid4().hex)
project = create_example(fldr)

# %%
network = project.network.build()


# %%
# We get our demand matrix from the project and create a computational view
proj_matrices = project.matrices
demand = proj_matrices.get_matrix("demand_omx")
demand.computational_view(["matrix"])

# %%
# Now let's perform our traffic assignment

assigclass = TrafficClass(
    name="car",
    mode="c",
    cost="free_flow_time",
    demand=demand,
    skimming=["free_flow_time", "distance"],
    block_centroid_flows=False,
)


assig = TrafficAssignment(
    network=network,
    classes=[assigclass],
    vdf=bpr,
    name_mapping={"alpha": 0.15, "beta": 4.0},
    time_field="free_flow_time",
    capacity_field="capacity",
    algorithm="bfw",
)
assig.execute(max_iterations=50, rgap=0.001)

# %%
#
with open(join(fldr, "aequilibrae.log")) as file:
    for idx, line in enumerate(file):
        print(idx + 1, "-", line)

# %%
# The log records each traffic class configuration and its demand totals.
# It then records the assignment configuration, including the VDF, algorithm,
# iteration limit, relative gap target, and thread count.
# Each iteration records its relative gap and step size. ``assig.report()``
# returns the same convergence data as a DataFrame.
# A new run records its configuration before the iteration messages.
