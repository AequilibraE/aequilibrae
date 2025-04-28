"""
.. _run_module_example:

Run module
==========

In this example we demonstrate how to use AequilibraE's run module using Sioux Falls example.
"""

# %%

# Imports

from uuid import uuid4
from tempfile import gettempdir
from os.path import join

from aequilibrae.utils.create_example import create_example

# %%
# Let's create the Sioux Falls example in an arbitrary folder.
folder = join(gettempdir(), uuid4().hex)

project = create_example(folder)

# %%
# First, let's check the matrix information using ``matrix_summary()``. This method
# provides us useful information such as the matrix total, mininum and maximum values
# in the array, and the number of non-empty pairs in the matrix.
# 
# Notice that the matrix summary is presented for each matrix core.
project.run.matrix_summary()

# %%
# If our matrices folder is empty, instead of a nested dictionary of data,
# AequilibraE run would return an empty dictionary.

# %%
# Let's create a graph for mode `car`.
mode = "c"

# %%
network = project.network
network.build_graphs(modes=[mode])
graph = network.graphs[mode]
graph.set_graph("distance")
graph.set_skimming("distance")
graph.set_blocked_centroid_flows(False)

# %%
# With the method `graph_summary()`, we can check the total number of links, nodes, and zones,
# as well as the compact number of links and nodes used for computation. If we had more than 
# one graph, its information would be displayed within the nested dictionary.

project.run.graph_summary()

# %%
