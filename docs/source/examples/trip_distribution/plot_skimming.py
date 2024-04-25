"""
.. _example_usage_skimming:

Network skimming
================

In this example, we show how to perform network skimming for Coquimbo, a city in La Serena Metropolitan Area in Chile.
"""

# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae.utils.create_example import create_example

# We create the example project inside our temp folder
fldr = join(gettempdir(), uuid4().hex)

project = create_example(fldr, "coquimbo")

# %%
import logging
import sys

# We the project opens, we can tell the logger to direct all messages to the terminal as well
logger = project.logger
stdout_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s;%(levelname)s ; %(message)s")
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

#%%
# Network Skimming
# ----------------

# %%
from aequilibrae.paths import NetworkSkimming
import numpy as np

# %%
# Let's build all graphs
project.network.build_graphs()
# We get warnings that several fields in the project are filled with NaNs.
# This is true, but we won't use those fields.

# %%
# We grab the graph for cars
graph = project.network.graphs["c"]

# we also see what graphs are available
project.network.graphs.keys()

# let's say we want to minimize the distance
graph.set_graph("distance")

# And will skim distance while we are at it, other fields like `free_flow_time` or `travel_time` can be added here as well
graph.set_skimming(["distance"])

# But let's say we only want a skim matrix for nodes 28-40, and 49-60 (inclusive), these happen to be a selection of western centroids.
graph.prepare_graph(np.array(list(range(28, 41)) + list(range(49, 91))))

# %%
# And run the skimming
skm = NetworkSkimming(graph)
skm.execute()

# %%

# The result is an AequilibraEMatrix object
skims = skm.results.skims

# Which we can manipulate directly from its temp file, if we wish
skims.matrices[:3, :3, :]

# %%

# Or access each matrix, lets just look at the first 3x3
skims.distance[:3, :3]

# %%

# We can save it to the project if we want
skm.save_to_project("base_skims")

# We can also retrieve this skim record to write something to its description
matrices = project.matrices
mat_record = matrices.get_record("base_skims")
mat_record.description = "minimized distance while also skimming distance for just a few nodes"
mat_record.save()

# %%
project.close()
