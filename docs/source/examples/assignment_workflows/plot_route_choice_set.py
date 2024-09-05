"""
.. _example_usage_route_choice_generation:

Route Choice set generation
===========================

In this example, we show how to generate route choice sets for estimation of route choice models, using a
a city in La Serena Metropolitan Area in Chile.
"""

# %%
# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
import numpy as np
from aequilibrae.utils.create_example import create_example

# sphinx_gallery_thumbnail_path = 'images/plot_route_choice_set.png'

# %%

# We create the example project inside our temp folder
fldr = join(gettempdir(), uuid4().hex)

project = create_example(fldr, "coquimbo")

# %%
# Model parameters
# ----------------

# %%
od_pairs_of_interest = [(71645, 79385), (77011, 74089)]
nodes_of_interest = (71645, 74089, 77011, 79385)

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

graph.set_graph("distance")

# We set the nodes of interest as centroids to make sure they are not simplified away when we create the network
graph.prepare_graph(np.array(nodes_of_interest))

# We allow flows through "centroid connectors" because our centroids are not really centroids
# If we have actual centroid connectors in the network (and more than one per centroid), then we
# should remove them from the graph
graph.set_blocked_centroid_flows(False)

# %%
# Route Choice class
# ------------------
# Here we'll construct and use the Route Choice class to generate our route sets
from aequilibrae.paths import RouteChoice

# %% 
# This object construct might take a minute depending on the size of the graph due to the construction of the
# compressed link to network link mapping that's required. This is a one time operation per graph and is cached. We
# need to supply a Graph and an AequilibraeMatrix or DataFrame via the ``add_demand`` method, if demand is not provided
# link loading cannot be preformed.
rc = RouteChoice(graph)

# %%
# Here we'll set the parameters of our set generation. There are two algorithms available: Link penalisation, and BFSLE
# based on the paper
# `"Route choice sets for very high-resolution data" <https://doi.org/10.1080/18128602.2012.671383>`_ 
# by Nadine Rieser-Schüssler, Michael Balmer & Kay W. Axhausen (2013).
#
# Our BFSLE implementation has been extended to allow applying link penalisation as well. Every
# link in all routes found at a depth are penalised with the `penalty` factor for the next depth. 
# So at a depth of 0 no links are penalised nor removed. At depth 1, all links found at depth 0 are penalised, 
# then the links marked for removal are removed. All links in the routes found at depth 1 are then penalised 
# for the next depth. The penalisation compounds. Pass set ``penalty=1.0`` to disable.
#
# It is highly recommended to set either ``max_routes`` or ``max_depth`` to prevent runaway results.

# rc.set_choice_set_generation("link-penalisation", max_routes=5, penalty=1.02)

# %%
# The 5% penalty (1.05) is likely a little too large, but it create routes that are distinct enough to make this simple
# example more interesting
rc.set_choice_set_generation("bfsle", max_routes=5, penalty=1.05)
rc.prepare(od_pairs_of_interest)
rc.execute(perform_assignment=True)
choice_set = rc.get_results().to_pandas()

# %%
# Plotting choice sets
# --------------------

# %%
# Now we will plot the paths we just created for the second OD pair
import folium
import geopandas as gpd

# %%
# Let's create a separate for each route so we can visualize one at a time
rlyr1 = folium.FeatureGroup("route 1")
rlyr2 = folium.FeatureGroup("route 2")
rlyr3 = folium.FeatureGroup("route 3")
rlyr4 = folium.FeatureGroup("route 4")
rlyr5 = folium.FeatureGroup("route 5")
od_lyr = folium.FeatureGroup("Origin and Destination")
layers = [rlyr1, rlyr2, rlyr3, rlyr4, rlyr5]

# %%
# We get the data we will use for the plot: Links, Nodes and the route choice set
links = gpd.GeoDataFrame(project.network.links.data, crs=4326)
nodes = gpd.GeoDataFrame(project.network.nodes.data, crs=4326)

plot_routes = choice_set[(choice_set["origin id"] == 77011)]["route set"].values

# Let's create the layers
colors = ["red", "blue", "green", "purple", "orange"]
for i, route in enumerate(plot_routes):
    rt = links[links.link_id.isin(route)]
    routes_layer = layers[i]
    for wkt in rt.geometry.to_wkt().values:
        points = wkt.replace("LINESTRING ", "").replace("(", "").replace(")", "").split(", ")
        points = "[[" + "],[".join([p.replace(" ", ", ") for p in points]) + "]]"
        # we need to take from x/y to lat/long
        points = [[x[1], x[0]] for x in eval(points)]

        _ = folium.vector_layers.PolyLine(points, color=colors[i], weight=4).add_to(routes_layer)

# Creates the points for both origin and destination
for i, row in nodes[nodes.node_id.isin((77011, 74089))].iterrows():
    point = (row.geometry.y, row.geometry.x)

    _ = folium.vector_layers.CircleMarker(
        point,
        popup=f"<b>link_id: {row.node_id}</b>",
        color="red",
        radius=5,
        fill=True,
        fillColor="red",
        fillOpacity=1.0,
    ).add_to(od_lyr)

# %%
# It is worthwhile to notice that using distance as the cost function, the routes are not the fastest ones as the
# freeway does not get used

# %%
# Create the map and center it in the correct place
long, lat = project.conn.execute("select avg(xmin), avg(ymin) from idx_links_geometry").fetchone()

map_osm = folium.Map(location=[lat, long], tiles="Cartodb Positron", zoom_start=12)
for routes_layer in layers:
    routes_layer.add_to(map_osm)
od_lyr.add_to(map_osm)
folium.LayerControl().add_to(map_osm)
map_osm

# %%
project.close()

# %%
# .. admonition:: References
# 
#   :ref:`route_choice`

# %%
# .. seealso::
#     The use of the following functions, methods, classes and modules is shown in this example:
#
#     * :func:`aequilibrae.paths.Graph` | :func:`aequilibrae.paths.RouteChoice`
#     * :func:`aequilibrae.matrix.AequilibraeMatrix`