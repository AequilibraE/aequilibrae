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
# sphinx_gallery_thumbnail_path = '../source/images/logs_image.png'

# %%

# We create an empty project on an arbitrary folder
fldr = join(gettempdir(), uuid4().hex)
project = create_example(fldr)

# %%
# We build our graphs
project.network.build_graphs()

graph = project.network.graphs["c"]
graph.set_graph("free_flow_time")
graph.set_skimming(["free_flow_time", "distance"])
graph.set_blocked_centroid_flows(False)

# %%
# We get our demand matrix from the project and create a computational view
proj_matrices = project.matrices
demand = proj_matrices.get_matrix("demand_omx")
demand.computational_view(["matrix"])

# %%
# Now let's perform our traffic assignment
assig = TrafficAssignment()

assigclass = TrafficClass(name="car", graph=graph, matrix=demand)

assig.add_class(assigclass)
assig.set_vdf("BPR")
assig.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
assig.set_capacity_field("capacity")
assig.set_time_field("free_flow_time")
assig.set_algorithm("bfw")
assig.max_iter = 50
assig.rgap_target = 0.001

assig.execute()

# %%
#
with open(join(fldr, "aequilibrae.log")) as file:
    for idx, line in enumerate(file):
        print(idx + 1, "-", line)

# %%
# In lines 1-7, we receive some warnings that our fields name and lane have ``NaN`` values.
# As they are not relevant to our example, we can move on.
#
# In lines 8-9 we get the Traffic Class specifications.
# We can see that there is only one traffic class (car). Its **graph** key presents information
# on blocked flow through centroids, number of centroids, links, and nodes.
# In the **matrix** key, we find information on where in the disk the matrix file is located.
# We also have information on the number of centroids and nodes, as well as on the matrix/matrices
# used for computation. In our example, we only have one matrix named matrix, and the total
# sum of this matrix element is equal to 360,600. If you have more than one matrix its data
# will be also displayed in the *matrix_cores* and *matrix_totals* keys.
#
# In lines 10-11 the log shows the Traffic Assignment specifications.
# We can see that the VDF parameters, VDF function, capacity and time fields, algorithm,
# maximum number of iterations, and target gap are just like the ones we set previously.
# The only information that might be new to you is the number of cores used for computation.
# If you haven't set any, AequilibraE is going to use the largest number of CPU threads
# available.
#
# Line 12 displays us a warning to indicate that AequilibraE is converting the data type
# of the cost field.
#
# Lines 13-61 indicate that we'll receive the outputs of a *bfw* algorithm.
# In the log there are also the number of the iteration, its relative gap, and the stepsize.
# The outputs in lines 15-60 are exactly the same as the ones provided by the function
# ``assig.report()``. Finally, the last line shows us that the *bfw* assignment has finished
# after 46 iterations because its gap is smaller than the threshold we configured (0.001).
#
# In case you execute a new traffic assignment using different classes or changing the
# parameters values, these new specification values would be stored in the log file as well
# so you can always keep a record of what you have been doing. One last reminder is that
# if we had created our project from OSM, the lines on top of the log would have been
# different to display information on the queries done to the server to obtain the data.
#
# Log image by `OSRS Wiki <https://oldschool.runescape.wiki/index.php?curid=66905#>`_
