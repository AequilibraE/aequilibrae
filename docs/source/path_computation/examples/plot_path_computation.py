"""
.. _example_usage_path_computation:

Path computation
================

In this example, we show how to perform path computation for Coquimbo, a city in La Serena Metropolitan Area in Chile.
"""
# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.paths.Graph`
#     * :func:`aequilibrae.paths.PathResults`

# %%

# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae.utils.create_example import create_example

# %%

# We create the example project inside our temp folder
fldr = join(gettempdir(), uuid4().hex)

project = create_example(fldr, "coquimbo")

# %%
import logging
import sys

# %%
# We the project opens, we can tell the logger to direct all messages to the terminal as well
logger = project.logger
stdout_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s;%(levelname)s ; %(message)s")
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

# %%
# Path Computation
# ----------------

# %%
# We build all graphs
project.network.build_graphs()
# We get warnings that several fields in the project are filled with ``NaN``s. 
# This is true, but we won't use those fields.

# %%
# We grab the graph for cars
graph = project.network.graphs["c"]

# we also see what graphs are available
project.network.graphs.keys()

# let's say we want to minimize the distance
graph.set_graph("distance")

# And will skim time and distance while we are at it
graph.set_skimming(["travel_time", "distance"])

# And we will allow paths to be computed going through other centroids/centroid connectors.
# We recommend you to `be extremely careful` with this setting.
graph.set_blocked_centroid_flows(False)

# %%
# Let's create a path results object from the graph and compute a path from 
# node 32343 (near the airport) to 22041 (near Fort Lambert, overlooking Coquimbo Bay).
res = graph.compute_path(32343, 22041)

# %%
# Computing paths directly from the graph is more straightforward, though we could
# alternatively use ``PathComputation`` class to achieve the same result.

# from aequilibrae.paths import PathResults

# res = PathResults()
# res.prepare(graph)
# res.compute_path(32343, 22041)

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
# Additionally, you can also provide ``early_exit=True`` or ``a_star=True`` to `compute_path` to 
# adjust its path-finding behavior.
# 
# Providing ``early_exit=True`` allows you to quit the path-finding procedure once it discovers 
# the destination. This setup works better for topographically close origin-destination pairs. 
# However, exiting early may cause subsequent calls to ``update_trace`` to recompute the tree 
# in cases where it typically wouldn't.
res = graph.compute_path(32343, 22041, early_exit=True)

# %%
# If you prefer to find a potentially non-optimal path to the destination faster, 
# provide ``a_star=True`` to use `A*` with a heuristic. This method always recomputes the 
# path's nodes, links, skims, and mileposts with ``update_trace``. 
# Note that a_star takes precedence over early_exit.
res = graph.compute_path(32343, 22041, a_star=True)

# %%
# If you are using `a_star`, it is possible to use different heuristics to compute the path. 
# By default, an equirectangular heuristic is used, and we can view the available heuristics via:
res.get_heuristics()

# %%
# If you prefer a more accurate but slower heuristic, you can choose "haversine", by setting:
res = graph.compute_path(32343, 22041, a_star=True, heuristic="haversine")

# %%
# Suppose you want to adjust the path to the University of La Serena instead of Fort Lambert. 
# It is possible to adjust the existing path computation for this alteration. The following code 
# allows both `early_exit` and `A*` settings to persist when calling ``update_trace``. If you’d 
# like to adjust them for subsequent path re-computations set the ``res.early_exit`` and 
# ``res.a_star`` attributes. Notice that this procedure is much faster when you have large networks.

res.a_star = False
res.update_trace(73131)

res.path_nodes

# %%
# If you want to show the path in Python.
# 
# We do NOT recommend this, though... It is very slow for real networks.
links = project.network.links.data.set_index("link_id")
links = links.loc[res.path]

# %%
links.explore(color="blue", style_kwds={'weight':5})

# %%
project.close()
