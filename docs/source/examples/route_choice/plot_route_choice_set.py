"""
.. _example_usage_route_choice_generation:

Route Choice set generation
===========================

In this example, we show how to generate route choice sets for estimation of route choice models, using a
a city in La Serena Metropolitan Area in Chile.
"""
# %%
# .. admonition:: References
# 
#   * :doc:`../../route_choice`

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.paths.route_choice`

# %%

# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join

import folium
import numpy as np
from aequilibrae.utils.create_example import create_example

# sphinx_gallery_thumbnail_path = '../source/_images/plot_route_choice_set.png'

# %%

# We create the example project inside our temp folder
fldr = join(gettempdir(), uuid4().hex)

project = create_example(fldr, "coquimbo")

# %%
# Model parameters
# ----------------
# Let's select a set of nodes of interest
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

# %%
# Route Choice class
# ------------------
# Here we'll construct and use the Route Choice class to generate our route sets
from aequilibrae.paths import RouteChoice

# %% 
# This object construct might take a minute depending on the size of the graph due to the construction of the
# compressed link to network link mapping that's required. This is a one time operation per graph and is cached.
rc = RouteChoice(graph)

# %%
# It is highly recommended to set either ``max_routes`` or ``max_depth`` to prevent runaway results.
# 
# We'll also set a 5% penalty (``penalty=1.05``), which is likely a little too large, but it creates routes that are 
# distinct enough to make this simple example more interesting.
rc.set_choice_set_generation("bfsle", max_routes=5, penalty=1.05)
rc.prepare(od_pairs_of_interest)
rc.execute(perform_assignment=True)

choice_set = rc.get_results()

# %%
# If we were interested in storing the route choice result, we could also write them to disk using the ``save_path_files`` method.

# rc.save_path_files(path)

# %%
# From those path files we could also preform a full assignment or select link analysis by using the ``execute_from_path_files`` method.

# rc.execute_from_path_files(path)

# %%
# Or if we had externally computed route choice sets, we can use AequilibraEs assignment procedures by 
# loading them with the ``execute_from_pandas`` method.

# rc.execute_from_pandas(path_files_df)

# %%
# Plotting choice sets
# --------------------

# %%
# Now we will plot the paths we just created for the second OD pair

# We get the data we will use for the plot: links, nodes and the route choice set
plot_routes = choice_set[(choice_set["origin id"] == 77011)]["route set"].values

links = project.network.links.data

# For ease of plot, we create a GeoDataFrame for each route in the choice set
route_1 = links[links.link_id.isin(np.absolute(plot_routes[0]))]
route_2 = links[links.link_id.isin(np.absolute(plot_routes[1]))]
route_3 = links[links.link_id.isin(np.absolute(plot_routes[2]))]
route_4 = links[links.link_id.isin(np.absolute(plot_routes[3]))]
route_5 = links[links.link_id.isin(np.absolute(plot_routes[4]))]

nodes = project.network.nodes.data
nodes = nodes[nodes["node_id"].isin([77011, 74089])]

# %%
map = route_1.explore(color="red", style_kwds={"weight": 3}, name="route_1")
map = route_2.explore(m=map, color="blue", style_kwds={"weight": 3}, name="route_2")
map = route_3.explore(m=map, color="green", style_kwds={"weight": 3}, name="route_3")
map = route_4.explore(m=map, color="purple", style_kwds={"weight": 3}, name="route_4")
map = route_5.explore(m=map, color="orange", style_kwds={"weight": 3}, name="route_5")

map = nodes.explore(m=map, color="black", style_kwds={"radius": 5, "fillOpacity": 1.0}, name="network_nodes")

folium.LayerControl().add_to(map)
map

# %%
project.close()
