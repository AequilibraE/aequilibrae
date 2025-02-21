"""
.. _example_usage_sub_area_analysis:

Route Choice with automated sub-area analysis
=============================================

In this example, we show how to perform sub-area analysis using route choice assignment, 
for a city in La Serena Metropolitan Area in Chile.

.. admonition:: References
 
   * :doc:`../../route_choice`

.. seealso::
    Several functions, methods, classes and modules are used in this example:

    * :func:`aequilibrae.paths.Graph`
    * :func:`aequilibrae.paths.RouteChoice`
    * :func:`aequilibrae.paths.SubAreaAnalysis`
    * :func:`aequilibrae.matrix.AequilibraeMatrix`
"""

# %%

# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join

import numpy as np
import folium

from aequilibrae.utils.create_example import create_example

# sphinx_gallery_thumbnail_path = '../source/_images/plot_subarea_analysis.png'

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
# Model parameters
# ----------------
# We'll set the parameters for our route choice model. These are the parameters that will be used to calculate the
# utility of each path. In our example, the utility is equal to :math:`distance * theta`,
# and the path overlap factor (PSL) is equal to :math:`beta`.

# Distance factor
theta = 0.011

# PSL parameter
beta = 1.1

# %%
# Let's build all graphs
project.network.build_graphs()
# We get warnings that several fields in the project are filled with NaNs.
# This is true, but we won't use those fields.

# %%
# We grab the graph for cars
graph = project.network.graphs["c"]

# %%
# We also see what graphs are available
project.network.graphs.keys()
# %%
# Let's say that utility is just a function of distance.
# So we build our *utility* field as the :math:`distance * theta`.
graph.network = graph.network.assign(utility=graph.network.distance * theta)

# %%
# Prepare the graph with all nodes of interest as centroids
graph.prepare_graph(graph.centroids)

# %%
# And set the cost of the graph the as the utility field just created
graph.set_graph("utility")

# %%
# We allow flows through "centroid connectors" because our centroids are not really centroids.
# If we have actual centroid connectors in the network (and more than one per centroid), then we
# should remove them from the graph.
graph.set_blocked_centroid_flows(False)
graph.graph.head()

# %%
# Mock demand matrix
# ------------------
# We'll create a mock demand matrix with demand ``10`` for every zone and prepare it for computation.
from aequilibrae.matrix import AequilibraeMatrix

names_list = ["demand"]

mat = AequilibraeMatrix()
mat.create_empty(zones=graph.num_zones, matrix_names=names_list, memory_only=True)
mat.index = graph.centroids[:]
mat.matrices[:, :, 0] = np.full((graph.num_zones, graph.num_zones), 10.0)
mat.computational_view()

# %%
# Sub-area preparation
# --------------------
# We need to define some polygon for out sub-area analysis, here we'll use a section of zones and create out polygon as
# the union of their geometry. It's best to choose a polygon that avoids any unnecessary intersections with links as
# the resource requirements of this approach grow quadratically with the number of links cut.
zones_of_interest = [29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 49, 50, 51, 52, 57, 58, 59, 60]
zones = project.zoning.data.set_index("zone_id")
zones = zones.loc[zones_of_interest]
zones.head()

# %%
# Sub-area analysis
# -----------------
# From here there are two main paths to conduct a sub-area analysis, manual or automated. AequilibraE ships with a small
# class that handle most of the details regarding the implementation and extract of the relevant data. It also exposes
# all the tools necessary to conduct this analysis yourself if you need fine grained control.

# %%
# Automated sub-area analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We first construct out SubAreaAnalysis object from the graph, zones, and matrix we previously constructed, then
# configure the route choice assignment and execute it. From there the ``post_process`` method is able to use the route
# choice assignment results to construct the desired demand matrix as a DataFrame.
from aequilibrae.paths import SubAreaAnalysis

subarea = SubAreaAnalysis(graph, zones, mat)
subarea.rc.set_choice_set_generation("lp", max_routes=3, penalty=1.02, store_results=False)
subarea.rc.execute(perform_assignment=True)
demand = subarea.post_process()
demand

# %%
# We'll re-prepare our graph but with our new "external" ODs.
new_centroids = np.unique(demand.reset_index()[["origin id", "destination id"]].to_numpy().reshape(-1))
graph.prepare_graph(new_centroids)
graph.set_graph("utility")
new_centroids

# %%
# We can then perform an assignment using our new demand matrix on the limited graph
from aequilibrae.paths import RouteChoice

rc = RouteChoice(graph)
rc.add_demand(demand)
rc.set_choice_set_generation("lp", max_routes=3, penalty=1.02, store_results=False, seed=123)
rc.execute(perform_assignment=True)

# %%
# Let's take the union of the zones GeoDataFrame as a polygon
poly = zones.union_all()
poly

# %%
# And plot the link loads for easy viewing
subarea_zone = folium.Polygon(
    locations=[(x[1], x[0]) for x in poly.boundary.coords],
    fill_color="blue",
    fill_opacity=0.1,
    fill=True,
    weight=1,
)

# %%
# Prepare our data for plotting
loads = rc.get_load_results()["demand"]
link_loads = loads[loads.tot > 0]
max_load = link_loads["tot"].max()
links = project.network.links.data
loaded_links = links.merge(link_loads, on="link_id", how="inner")
factor = 10 / max_load

# %%
m = loaded_links.explore(
    color="red",
    style_kwds={
        "style_function": lambda x: {
            "weight": x["properties"]["tot"] * factor,
        }
    },
)

subarea_zone.add_to(m)
m

# %%
# Sub-area further preparation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# It's useful later on to know which links from the network cross our polygon.
links = project.network.links.data
inner_links = links[links.crosses(poly.boundary)].sort_index()
inner_links.head()

# %%
# Let's filter those network links to graph links, dropping any dead ends and creating a `link_id`,
# `dir` multi-index.
g = (
    graph.graph.set_index("link_id")
    .loc[inner_links.link_id]
    .drop(graph.dead_end_links, errors="ignore")
    .reset_index()
    .set_index(["link_id", "direction"])
)
g.head()

# %%
# Here we'll quickly visualise what our sub-area is looking like.
# We'll plot the polygon from our zoning system and the links that it cuts.
m = inner_links.explore(color="red", style_kwds={"weight": 4})
subarea_zone.add_to(m)
m
