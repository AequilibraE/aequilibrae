"""
.. _example_usage_paths:

Path and skimming
=================

On this example we show how to perform path computation and network skimming
for the Sioux Falls example model.
"""

# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae.utils.create_example import create_example

# We create the example project inside our temp folder
fldr = join(gettempdir(), uuid4().hex)

project = create_example(fldr)

# %%
import logging
import sys

# We the project open, we can tell the logger to direct all messages to the terminal as well
logger = project.logger
stdout_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s;%(levelname)s ; %(message)s")
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

#%%
# Path Computation
# ----------------

# %%
from aequilibrae.paths import PathResults

# %%
# we build all graphs
project.network.build_graphs()
# We get warnings that several fields in the project are filled with NaNs.  Which is true, but we won't use those fields

# %%
# we grab the graph for cars
graph = project.network.graphs["c"]

# we also see what graphs are available
# project.network.graphs.keys()

# let's say we want to minimize distance
graph.set_graph("distance")

# And will skim time and distance while we are at it
graph.set_skimming(["free_flow_time", "distance"])

# And we will allow paths to be compute going through other centroids/centroid connectors
# required for the Sioux Falls network, as all nodes are centroids
# BE CAREFUL WITH THIS SETTING
graph.set_blocked_centroid_flows(False)

# %%
# instantiate a path results object and prepare it to work with the graph
res = PathResults()
res.prepare(graph)

# compute a path from node 8 to 13
res.compute_path(8, 4)

# %%

# We can get the sequence of nodes we traverse
res.path_nodes

# %%

# We can get the link sequence we traverse
res.path

# %%

# We can get the mileposts for our sequence of nodes
res.milepost

# %%

# If we want to compute the path for a different destination and same origin, we can just do this
# It is way faster when you have large networks
res.update_trace(13)

# %%

res.path_nodes

# %%
# If you want to show the path in Python
# We do NOT recommend this, though....  It is very slow for real networks
import matplotlib.pyplot as plt
from shapely.ops import linemerge

links = project.network.links

# We plot the entire network
curr = project.conn.cursor()
curr.execute("Select link_id from links;")

for lid in curr.fetchall():
    geo = links.get(lid[0]).geometry
    plt.plot(*geo.xy, color="red")

path_geometry = linemerge(links.get(lid).geometry for lid in res.path)
plt.plot(*path_geometry.xy, color="blue", linestyle="dashed", linewidth=2)
plt.show()

#%%
# Now to skimming
# ---------------

# %%
from aequilibrae.paths import NetworkSkimming

# %%
# But let's say we only want a skim matrix for nodes 1, 3, 6 & 8
import numpy as np

graph.prepare_graph(np.array([1, 3, 6, 8]))
# %%

# And run the skimming
skm = NetworkSkimming(graph)
skm.execute()

# %%
# The result is an AequilibraEMatrix object
skims = skm.results.skims

# Which we can manipulate directly from its temp file, if we wish
skims.matrices

# %%

# Or access each matrix
skims.free_flow_time

# %%

# We can save it to the project if we want
skm.save_to_project("base_skims")

# We can also retrieve this skim record to write something to its description
matrices = project.matrices
mat_record = matrices.get_record("base_skims")
mat_record.description = "minimized FF travel time while also skimming distance for just a few nodes"
mat_record.save()

# %%
project.close()
