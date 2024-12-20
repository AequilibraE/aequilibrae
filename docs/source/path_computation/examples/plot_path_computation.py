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
# Additionally we could also provide ``early_exit=True`` or ``a_star=True`` to ``compute_path``
# to adjust its path finding behaviour. Providing ``early_exit=True`` will allow the path finding 
# to quit once it's discovered the destination, this means it will perform better for ODs that are 
# topographically close. However, exiting early may cause subsequent calls to ``update_trace``
# to recompute the tree in cases where it usually wouldn't. ``a_star=True`` has precedence of ``early_exit=True``.
res = graph.compute_path(32343, 22041, early_exit=True)

# %%
# If you'd prefer to find a potentially non-optimal path to the destination faster provide 
# ``a_star=True`` to use `A*` with a heuristic. 
# With this method ``update_trace`` will always recompute the path.
res = graph.compute_path(32343, 22041, a_star=True)

# %%
# By default a equirectangular heuristic is used. We can view the available heuristics via
res.get_heuristics()

# %%
# If you'd like the more accurate, but slower, but more accurate haversine heuristic you can set it using
res = graph.compute_path(32343, 22041, a_star=True, heuristic="haversine")

# %%
# If we want to compute the path for a different destination and the same origin, we can just do this.
# It is way faster when you have large networks.
# Here we'll adjust our path to the University of La Serena. 
# Our previous early exit and `A*` settings will persist with calls to ``update_trace``. 
# If you'd like to adjust them for subsequent path re-computations set the ``res.early_exit`` 
# and ``res.a_star`` attributes.
res.a_star = False
res.update_trace(73131)

# %%
res.path_nodes

# %%
# If you want to show the path in Python.
# 
# We do NOT recommend this, though... It is very slow for real networks.
import matplotlib.pyplot as plt
from shapely.ops import linemerge

# %%
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

# %%
project.close()
