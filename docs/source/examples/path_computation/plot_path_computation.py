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
#     * :func:`aequilibrae.paths.graph`
#     * :func:`aequilibrae.paths.results.path_results`

# %%
# Imports
from os.path import join
from tempfile import gettempdir
from uuid import uuid4

import geopandas as gpd
import pandas as pd

from aequilibrae.paths import estimate_heuristic_scale
from aequilibrae.utils.create_example import create_example
from aequilibrae.utils.logging_utils import basic_config

# %%
# We create the example project inside our temp folder
fldr = join(gettempdir(), uuid4().hex)

project = create_example(fldr, "coquimbo")

# %%

# We'll also apply a basic logging configuration.

basic_config()

# %%
# Path Computation
# ----------------

# %%
# We build all graphs
project.network.build_graphs()
# We get warnings that several fields in the project are filled with ``NaN``s.
# This is true, but we won't use those fields.

# %%
# We grab the graph for cars,
graph = project.network.graphs["c"]

# %%
# we'll also see what graphs are available.
project.network.graphs.keys()

# %%
# Let's say we want to minimise the distance,
graph.set_graph("distance")

# %%
# and will skim time and distance while we are at it.
graph.set_skimming(["travel_time", "distance"])

# %%
# Let's create a path results object from the graph and compute a path from
# node 32343 (near the airport) to 22041 (near Fort Lambert, overlooking Coquimbo Bay).
res = graph.compute_path(32343, 22041)

# %%
# Computing paths directly from the graph is more straightforward, though we could
# alternatively use the ``PathResults`` class to achieve the same result.
#
# from aequilibrae.paths import PathResults
# res = PathResults(graph, 32343, 22041)
#
# ``graph.compute_path`` has already performed this setup above.

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
# Additionally, you can also provide ``early_exit=True`` or ``a_star=True`` to `compute_path` to adjust its
# path-finding behaviour.
#
# Providing ``early_exit=True`` allows you to quit the path-finding procedure once it discovers the destination. This
# setup works better for topographically close origin-destination pairs.  However, exiting early may cause subsequent
# calls to ``update_trace`` to recompute the tree in cases where it typically wouldn't.
res = graph.compute_path(32343, 22041, early_exit=True)

# %%
# To guide the search towards the destination, provide ``a_star=True`` to use `A*` with a heuristic. ``update_trace``
# reuses finalised paths, or searches again if needed, retaining the algorithm, heuristic, scale and heap.  Note that
# a_star takes precedence over early_exit. Euclidean distance needs node coordinates projected to a suitable local CRS
# and a scale parameter to transform it into a cost value.
lonlat = graph.lonlat_index
points = gpd.GeoSeries(gpd.points_from_xy(lonlat.lon, lonlat.lat), index=lonlat.index, crs=4326)
points = points.to_crs(points.estimate_utm_crs())
coordinates = points.get_coordinates()

scale = estimate_heuristic_scale(graph, coordinates)
res = graph.compute_path(32343, 22041, a_star=True, coordinates=coordinates, heuristic_scale=scale)

# %%
# If you are using `a_star`, it is possible to use different heuristics to compute the path.
# By default, a Euclidean heuristic is used, and we can view the available heuristics via:
res.get_heuristics()

# %%
# To use geographic distance without projecting the coordinates, choose "haversine".  It uses the graph's
# longitude/latitude data and needs its own scale.
scale = estimate_heuristic_scale(graph, heuristic="haversine")
res = graph.compute_path(32343, 22041, a_star=True, heuristic="haversine", heuristic_scale=scale)

# %%
# Suppose you want to adjust the path to the University of La Serena instead of Fort Lambert.  It is possible to adjust
# the existing path computation for this alteration. The following code allows both `early_exit` and `A*` settings to
# persist when calling ``update_trace``. If you’d like to adjust them for subsequent path re-computations, call
# ``compute_path`` with the desired settings. Notice that this procedure is much faster when you have large networks and
# the search for the previous destination happened to include the new destination.

res.update_trace(73131)

res.path_nodes

# %%
# If you want to show the path in Python.
#
# We do NOT recommend this, though... It is very slow for real networks.
links = project.network.links.data.set_index("link_id")
links = links.loc[res.path]

# %%
links.explore(color="blue", style_kwds={"weight": 5})

# %%
project.close()
