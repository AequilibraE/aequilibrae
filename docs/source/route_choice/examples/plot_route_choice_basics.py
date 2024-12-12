"""
.. _example_usage_route_choice:

Route Choice
============

In this example, we show how to perform route choice set generation using BFSLE and Link penalisation, for a city in La
Serena Metropolitan Area in Chile.
"""
# %%
# .. admonition:: References
# 
#   * :ref:`route_choice`

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.paths.Graph`
#     * :func:`aequilibrae.paths.RouteChoice`
#     * :func:`aequilibrae.matrix.AequilibraeMatrix`

# %%

# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae.utils.create_example import create_example

# sphinx_gallery_thumbnail_path = '../source/images/plot_route_choice_assignment.png'

# %%

# We create the example project inside our temp folder
fldr = join(gettempdir(), uuid4().hex)

project = create_example(fldr, "coquimbo")

# %%
import logging
import sys

# %%

# When the project opens, we can tell the logger to direct all messages to the terminal as well
logger = project.logger
stdout_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s;%(levelname)s ; %(message)s")
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

# %%
# Model parameters
# ----------------

import numpy as np

# %%
# We'll set the parameters for our route choice model. These are the parameters that will be used to calculate the
# utility of each path. In our example, the utility is equal to :math:`distance * theta`,
# and the path overlap factor (PSL) is equal to :math:`beta`.

# Distance factor
theta = 0.00011

# PSL parameter
beta = 1.1

# %%
# Let's select a set of nodes of interest
nodes_of_interest = (71645, 74089, 77011, 79385)

# %%
# Let's build all graphs
project.network.build_graphs()
# We get warnings that several fields in the project are filled with NaNs.
# This is true, but we won't use those fields.

# %%
# We also see what graphs are available
project.network.graphs.keys()

# %%
# We grab the graph for cars
graph = project.network.graphs["c"]

# Let's say that utility is just a function of distance, so we build our 'utility' field as distance * theta
graph.network = graph.network.assign(utility=graph.network.distance * theta)

# Prepare the graph with all nodes of interest as centroids
graph.prepare_graph(np.array(nodes_of_interest))

# And set the cost of the graph the as the utility field just created
graph.set_graph("utility")

# We allow flows through "centroid connectors" because our centroids are not really centroids
# If we have actual centroid connectors in the network (and more than one per centroid), then we
# should remove them from the graph
graph.set_blocked_centroid_flows(False)

# %%
# Mock demand matrix
# ------------------
# We'll create a mock demand matrix with demand 1 for every zone and prepare it for computation.
from aequilibrae.matrix import AequilibraeMatrix

names_list = ["demand", "5x demand"]

mat = AequilibraeMatrix()
mat.create_empty(zones=graph.num_zones, matrix_names=names_list, memory_only=True)
mat.index = graph.centroids[:]
mat.matrices[:, :, 0] = np.full((graph.num_zones, graph.num_zones), 10.0)
mat.matrices[:, :, 1] = np.full((graph.num_zones, graph.num_zones), 50.0)
mat.computational_view()

# %%
# Create plot function
# --------------------
# Before dive into the Route Choice class, let's define a function to plot assignment results.
import folium

# %%
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

# %%
# Route Choice class
# ------------------
# Here we'll construct and use the Route Choice class to generate our route sets
from aequilibrae.paths import RouteChoice

# %%
# This object construct might take a minute depending on the size of the graph due to the construction of the compressed
# link to network link mapping that's required. This is a one time operation per graph and is cached.
rc = RouteChoice(graph)

# Let's check the default parameters for the Route Choice class
print(rc.default_parameters)

# %%
# Let's add the demand. If it's not provided, link loading cannot be preformed.
rc.add_demand(mat)

# %%
# It is highly recommended to set either ``max_routes`` or ``max_depth`` to prevent runaway results.
rc.set_choice_set_generation("bfsle", max_routes=5)

# %%
# We can now perform a computation for single OD pair if we'd like. Here we do one between the first and last centroid
# as well as an assignment.
results = rc.execute_single(77011, 74089, demand=1.0)
print(results[0])

# %%
# Because we asked it to also perform an assignment we can access the various results from that.
# The default return is a Pyarrow Table but Pandas is nicer for viewing.
res = rc.get_results().to_pandas()
res.head()

# %%
plot_results(rc.get_load_results()["demand"])

# %%
# Batch operations
# ----------------
# To perform a batch operation we need to prepare the object first. We can either provide a list of tuple of the OD
# pairs we'd like to use, or we can provided a 1D list and the generation will be run on all permutations.
rc.prepare()

# %%
# Now we can perform a batch computation with an assignment
rc.execute(perform_assignment=True)
res = rc.get_results().to_pandas()
res.head()

# %%
# Since we provided a matrix initially we can also perform link loading based on our assignment results.
rc.get_load_results()

# %% 
# We can plot these as well
plot_results(rc.get_load_results()["demand"])

# %%
# Select link analysis
# --------------------
# We can also enable select link analysis by providing the links and the directions that we are interested in. Here we
# set the select link to trigger when (7369, 1) and (20983, 1) is utilised in "sl1" and "sl2" when (7369, 1) is
# utilised.
rc.set_select_links({"sl1": [[(7369, 1), (20983, 1)]], "sl2": [[(7369, 1)]]})
rc.execute(perform_assignment=True)

# %%
# We can get then the results in a Pandas DataFrame for both the network.
sl = rc.get_select_link_loading_results()
sl

# %%
# We can also access the OD matrices for this link loading. These matrices are sparse and can be converted to
# SciPy sparse matrices for ease of use. They're stored in a dictionary where the key is the matrix name concatenated
# with the select link set name via an underscore.
rc.get_select_link_od_matrix_results()

# %%
od_matrix = rc.get_select_link_od_matrix_results()["sl1"]["demand"]
od_matrix.to_scipy().toarray()

# %%
project.close()
