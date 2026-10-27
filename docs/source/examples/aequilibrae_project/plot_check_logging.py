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
network = project.network


# %%
# We get our demand matrix from the project and create a computational view
proj_matrices = project.matrices
demand = proj_matrices.get_matrix("demand_omx", subset=["matrix"])

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
# The log records the Traffic Class specifications, including the network mode,
# cost field, PCE, value of time, and matrix totals.
#
# The log then shows the Traffic Assignment specifications.
# We can see that the VDF parameters, VDF function, capacity and time fields, algorithm,
# maximum number of iterations, and target gap are just like the ones we set previously.
# The only information that might be new to you is the number of cores used for computation.
# If you haven't set any, AequilibraE is going to use the largest number of CPU threads
# available.
#
# Each iteration records its relative gap and step size. ``assig.report()``
# returns the same convergence data as a DataFrame.
#
# In case you execute a new traffic assignment using different classes or changing the
# parameters values, these new specification values would be stored in the log file as well
# so you can always keep a record of what you have been doing. One last reminder is that
# if we had created our project from OSM, the lines on top of the log would have been
# different to display information on the queries done to the server to obtain the data.
#
# Log image by `OSRS Wiki <https://oldschool.runescape.wiki/index.php?curid=66905#>`_
