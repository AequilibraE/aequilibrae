"""
.. _example_usage_route_choice:

Route Choice
=================

In this example, we show how to perform route choice set generation using BFSLE and Link penalisation, for a city in La Serena Metropolitan Area in Chile.
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

# %%
# Route Choice
# ---------------

# %%
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

# But let's say we only want a skim matrix for nodes 28-40, and 49-60 (inclusive), these happen to be a selection of western centroids.
graph.prepare_graph(np.array(list(range(28, 41)) + list(range(49, 91))))

# %%
# Mock demand matrix
# ~~~~~~~~~~~~~~~~~~
# We'll create a mock demand matrix with demand `1` for every zone.
from aequilibrae.matrix import AequilibraeMatrix

names_list = ["demand", "5x demand"]

mat = AequilibraeMatrix()
mat.create_empty(zones=graph.num_zones, matrix_names=names_list, memory_only=True)
mat.index = graph.centroids[:]
mat.matrices[:, :, 0] = np.full((graph.num_zones, graph.num_zones), 1.0)
mat.matrices[:, :, 1] = np.full((graph.num_zones, graph.num_zones), 5.0)
mat.computational_view()

# %%
# Route Choice class
# ~~~~~~~~~~~~~~~~~~
# Here we'll construct and use the Route Choice class to generate our route sets
from aequilibrae.paths import RouteChoice

# %%
# This object construct might take a minute depending on the size of the graph due to the construction of the compressed link to network link mapping that's required.
# This is a one time operation per graph and is cached.
# We need to supply a Graph and optionally a AequilibraeMatrix, if the matrix is not provided link loading cannot be preformed.
rc = RouteChoice(graph, mat)

# %%
# Here we'll set the parameters of our set generation. There are two algorithms available: Link penalisation, or BFSLE based on the paper
# "Route choice sets for very high-resolution data" by Nadine Rieser-Schüssler, Michael Balmer & Kay W. Axhausen (2013).
# https://doi.org/10.1080/18128602.2012.671383
# It is highly recommended to set either `max_routes` or `max_depth` to prevent runaway results.

# rc.set_choice_set_generation("link-penalisation", max_routes=5, penalty=1.1)
rc.set_choice_set_generation("bfsle", max_routes=5, beta=1.1, theta=1.1)

# %%
# All parameters are optional, the defaults are:
print(rc.default_paramaters)

# %%
# We can now perform a computation for single OD pair if we'd like. Here we do one between the first and last centroid as well an an assignment.
results = rc.execute_single(28, 90, perform_assignment=True)
print(results[0])

# %%
# Because we asked it to also perform an assignment we can access the various results from that
# The default return is a Pyarrow Table but Pandas is nicer for viewing.
rc.get_results().to_pandas()

# %%
# To perform a batch operation we need to prepare the object first. We can either provide a list of tuple of the OD pairs we'd like to use, or we can provided a 1D list
# and the generation will be run on all permutations.
rc.prepare(graph.centroids[:5])  # You can inspect the result with rc.nodes

# %%
# Now we can perform a batch computation with an assignment
rc.execute(perform_assignment=True)
rc.get_results().to_pandas()

# %%
# Since we provided a matrix initially we can also perform link loading based on our assignment results.
# We can specify which link loading we want, either just uncompressed, just compressed, or both.
rc.get_load_results(which="both")

# %%
project.close()
