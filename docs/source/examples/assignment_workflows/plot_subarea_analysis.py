"""
.. _example_usage_sub_area_analysis:

Route Choice with sub-area analysis
===================================

In this example, we show how to perform sub-area analysis using route choice assignment, for a city in La Serena
Metropolitan Area in Chile.

.. admonition:: References
 
   * :ref:`route_choice`

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
import itertools

import pandas as pd
import numpy as np
import folium

from aequilibrae.utils.create_example import create_example

# sphinx_gallery_thumbnail_path = '../source/images/plot_subarea_analysis.png'

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
subarea.rc.set_choice_set_generation("lp", max_routes=5, penalty=1.02, store_results=False)
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
rc.set_choice_set_generation("link-penalisation", max_routes=5, penalty=1.02, store_results=False, seed=123)
rc.execute(perform_assignment=True)

# %%
# And plot the link loads for easy viewing
subarea_zone = folium.Polygon(
    locations=[(x[1], x[0]) for x in zones.unary_union.boundary.coords],
    fill_color="blue",
    fill_opacity=0.5,
    fill=True,
    stroke=False,
)

def plot_results(link_loads):
    link_loads = link_loads[link_loads.tot > 0]
    max_load = link_loads["tot"].max()
    links = project.network.links.data
    loaded_links = links.merge(link_loads, on="link_id", how="inner")

    loads_lyr = folium.FeatureGroup("link_loads")

    # Maximum thickness we would like is probably a 10, so let's make sure we don't go over that
    factor = 10 / max_load

    # Let's create the layers
    for _, rec in loaded_links.iterrows():
        points = rec.geometry.wkt.replace("LINESTRING ", "").replace("(", "").replace(")", "").split(", ")
        points = "[[" + "],[".join([p.replace(" ", ", ") for p in points]) + "]]"
        # we need to take from x/y to lat/long
        points = [[x[1], x[0]] for x in eval(points)]
        _ = folium.vector_layers.PolyLine(
            points,
            tooltip=f"link_id: {rec.link_id}, Flow: {rec.tot:.3f}",
            color="red",
            weight=factor * rec.tot,
        ).add_to(loads_lyr)
    long, lat = project.conn.execute("select avg(xmin), avg(ymin) from idx_links_geometry").fetchone()

    map_osm = folium.Map(location=[lat, long], tiles="Cartodb Positron", zoom_start=12)
    loads_lyr.add_to(map_osm)
    folium.LayerControl().add_to(map_osm)
    return map_osm


map = plot_results(rc.get_load_results()["demand"])
subarea_zone.add_to(map)
map

# %%
# Sub-area further preparation
# ````````````````````````````

# %%
# We take the union of this GeoDataFrame as our polygon.
poly = zones.union_all()
poly

# %%
# It's useful later on to know which links from the network cross our polygon.
links = project.network.links.data
inner_links = links[links.crosses(poly.boundary)].sort_index()
inner_links.head()

# %%
# As well as which nodes are interior.
nodes = project.network.nodes.data.set_index("node_id")
inside_nodes = nodes.sjoin(zones, how="inner").sort_index()
inside_nodes.head()

# %%
# Here we filter those network links to graph links, dropping any dead ends and creating a `link_id, dir` multi-index.
g = (
    graph.graph.set_index("link_id")
    .loc[inner_links.link_id]
    .drop(graph.dead_end_links, errors="ignore")
    .reset_index()
    .set_index(["link_id", "direction"])
)
g.head()

# %%
# Sub-area visualisation
# ``````````````````````

# %%
# Here we'll quickly visualise what out sub-area is looking like. We'll plot the polygon from our zoning system and the
# links that it cuts.
points = [(link_id, list(x.coords)) for link_id, x in zip(inner_links.link_id, inner_links.geometry)]
subarea_layer = folium.FeatureGroup("Cut links")

for link_id, line in points:
    _ = folium.vector_layers.PolyLine(
        [(x[1], x[0]) for x in line],
        tooltip=f"link_id: {link_id}",
        color="red",
    ).add_to(subarea_layer)

long, lat = project.conn.execute("select avg(xmin), avg(ymin) from idx_links_geometry").fetchone()

map_osm = folium.Map(location=[lat, long], tiles="Cartodb Positron", zoom_start=12)

subarea_zone.add_to(map_osm)

subarea_layer.add_to(map_osm)
_ = folium.LayerControl().add_to(map_osm)
map_osm

# %%
# Manual sub-area analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~

# %%
# In order to perform out analysis we need to know what OD pairs have flow that enters and/or exists our polygon. To do
# so we perform a select link analysis on all links and pairs of links that cross the boundary. We create them as
# tuples of tuples to make represent the select link AND sets.
edge_pairs = {x: (x,) for x in itertools.permutations(g.index, r=2)}
single_edges = {x: ((x,),) for x in g.index}
f"Created: {len(edge_pairs)} edge pairs from {len(single_edges)} edges"

# %%
# Here we'll construct and use the Route Choice class to generate our route sets
from aequilibrae.paths import RouteChoice

# %%
# We'll re-prepare out graph quickly
project.network.build_graphs()
graph = project.network.graphs["c"]
graph.network = graph.network.assign(utility=graph.network.distance * theta)
graph.prepare_graph(graph.centroids)
graph.set_graph("utility")
graph.set_blocked_centroid_flows(False)

# %%
# This object construction might take a minute depending on the size of the graph due to the construction of the
# compressed link to network link mapping that's required. This is a one time operation per graph and is cached. We
# need to supply a Graph and an AequilibraeMatrix or DataFrame via the ``add_demand`` method, if demand is not provided
# link loading cannot be preformed.
rc = RouteChoice(graph)
rc.add_demand(mat)

# %%
# Here we add the union of edges as select link sets.
rc.set_select_links(single_edges | edge_pairs)

# %%
# For the sake of demonstration we limit out demand matrix to a few OD pairs. This filter is also possible with the
# automated approach, just edit the ``subarea.rc.demand.df`` DataFrame, however make sure the index remains intact.
ods_pairs_of_interest = [
    (4, 39),
    (92, 37),
    (31, 58),
    (4, 19),
    (39, 34),
]
ods_pairs_of_interest = ods_pairs_of_interest + [(x[1], x[0]) for x in ods_pairs_of_interest]
rc.demand.df = rc.demand.df.loc[ods_pairs_of_interest].sort_index().astype(np.float32)
rc.demand.df

# %%
# Perform the assignment
rc.set_choice_set_generation("link-penalisation", max_routes=5, penalty=1.02, store_results=False, seed=123)
rc.execute(perform_assignment=True)


# %%
# We can visualise the current links loads
map = plot_results(rc.get_load_results()["demand"])
subarea_zone.add_to(map)
map

# %%
# We'll pull out just OD matrix results as well we need it for the post-processing, we'll also convert the sparse
# matrices to SciPy COO matrices.
sl_od = rc.get_select_link_od_matrix_results()
edge_totals = {k: sl_od[k]["demand"].to_scipy() for k in single_edges}
edge_pair_values = {k: sl_od[k]["demand"].to_scipy() for k in edge_pairs}

# %%
# For the post processing, we are interested in the demand of OD pairs that enter or exit the sub-area, or do both. For
# the single enters and exists we can extract that information from the single link select link results. We also need to
# map the links that cross the boundary to the origin/destination node and the node that appears on the outside of the
# sub-area.
from collections import defaultdict

entered = defaultdict(float)
exited = defaultdict(float)
for (link_id, dir), v in edge_totals.items():
    link = g.loc[link_id, dir]
    for (o, d), load in v.todok().items():
        o = graph.all_nodes[o]
        d = graph.all_nodes[d]

        o_inside = o in inside_nodes.index
        d_inside = d in inside_nodes.index

        if o_inside and not d_inside:
            exited[o, graph.all_nodes[link.b_node]] += load
        elif not o_inside and d_inside:
            entered[graph.all_nodes[link.a_node], d] += load
        elif not o_inside and not d_inside:
            pass


# %%
# Here he have the load that entered the sub-area
entered

# %%
# and the load that exited the sub-area
exited

# %%
# To find the load that both entered and exited we can look at the edge pair select link results.
through = defaultdict(float)
for (l1, l2), v in edge_pair_values.items():
    link1 = g.loc[l1]
    link2 = g.loc[l2]

    for (o, d), load in v.todok().items():
        o_inside = o in inside_nodes.index
        d_inside = d in inside_nodes.index

        if not o_inside and not d_inside:
            through[graph.all_nodes[link1.a_node], graph.all_nodes[link2.b_node]] += load

through

# %%
# With these results we can construct a new demand matrix. Usually this would be now transplanted onto another network,
# however for demonstration purposes we'll reuse the same network.
demand = pd.DataFrame(
    list(entered.values()) + list(exited.values()) + list(through.values()),
    index=pd.MultiIndex.from_tuples(
        list(entered.keys()) + list(exited.keys()) + list(through.keys()), names=["origin id", "destination id"]
    ),
    columns=["demand"],
).sort_index()
demand.head()

# %%
# We'll re-prepare our graph but with our new "external" ODs.
new_centroids = np.unique(demand.reset_index()[["origin id", "destination id"]].to_numpy().reshape(-1))
graph.prepare_graph(new_centroids)
graph.set_graph("utility")
new_centroids

# %%
# Re-perform our assignment
rc = RouteChoice(graph)
rc.add_demand(demand)
rc.set_choice_set_generation("link-penalisation", max_routes=5, penalty=1.02, store_results=False, seed=123)
rc.execute(perform_assignment=True)

# %%
# And plot the link loads for easy viewing
map = plot_results(rc.get_load_results()["demand"])
subarea_zone.add_to(map)
map
