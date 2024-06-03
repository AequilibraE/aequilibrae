""".. _example_usage_route_choice:

Route Choice
=================

In this example, we show how to perform route choice set generation using BFSLE and Link penalisation, for a city in La
Serena Metropolitan Area in Chile.

"""

# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae.utils.create_example import create_example
# sphinx_gallery_thumbnail_path = 'images/plot_route_choice_assignment.png'

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
# Model parameters
# ~~~~~~~~~~~~~~~~
# We'll set the parameters for our route choice model. These are the parameters that will be used to calculate the
# utility of each path. In our example, the utility is equal to *theta* * distance
# And the path overlap factor (PSL) is equal to *beta*.

# Distance factor
theta=0.011

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

# we also see what graphs are available
project.network.graphs.keys()

od_pairs_of_interest = [(71645, 79385), (77011, 74089)]
nodes_of_interest = (71645, 74089, 77011, 79385)

# let's say that utility is just a function of distance
# So we build our *utility* field as the distance times theta
graph.network = graph.network.assign(utility=graph.network.distance * theta)

# Prepare the graph with all nodes of interest as centroids
graph.prepare_graph(np.array(nodes_of_interest))

# And set the cost of the graph the as the utility field just created
graph.set_graph("utility")

# %%
# Mock demand matrix
# ~~~~~~~~~~~~~~~~~~
# We'll create a mock demand matrix with demand `1` for every zone.
from aequilibrae.matrix import AequilibraeMatrix

names_list = ["demand", "5x demand"]

mat = AequilibraeMatrix()
mat.create_empty(zones=graph.num_zones, matrix_names=names_list, memory_only=True)
mat.index = graph.centroids[:]
mat.matrices[:, :, 0] = np.full((graph.num_zones, graph.num_zones), 10.0)
mat.matrices[:, :, 1] = np.full((graph.num_zones, graph.num_zones), 50.0)
mat.computational_view()

# %%
# Route Choice class
# ~~~~~~~~~~~~~~~~~~
# Here we'll construct and use the Route Choice class to generate our route sets
from aequilibrae.paths import RouteChoice

# %%
# This object construct might take a minute depending on the size of the graph due to the construction of the compressed
# link to network link mapping that's required.  This is a one time operation per graph and is cached. We need to
# supply a Graph and optionally a AequilibraeMatrix, if the matrix is not provided link loading cannot be preformed.
rc = RouteChoice(graph, mat)

# %%

# Here we'll set the parameters of our set generation. There are two algorithms available: Link penalisation, or BFSLE
# based on the paper
# "Route choice sets for very high-resolution data" by Nadine Rieser-Schüssler, Michael Balmer & Kay W. Axhausen (2013).
# https://doi.org/10.1080/18128602.2012.671383
#
# Our BFSLE implementation is slightly different and has extended to allow applying link penalisation as well. Every
# link in all routes found at a depth are penalised with the `penalty` factor for the next depth. So at a depth of 0 no
# links are penalised nor removed. At depth 1, all links found at depth 0 are penalised, then the links marked for
# removal are removed. All links in the routes found at depth 1 are then penalised for the next depth. The penalisation
# compounds. Pass set `penalty=1.0` to disable.
#
# To assist in filtering out bad results during the assignment, a `cutoff_prob` parameter can be provided to exclude
# routes from the path-sized logit model. The `cutoff_prob` is used to compute an inverse binary logit and obtain a max
# difference in utilities. If a paths total cost is greater than the minimum cost path in the route set plus the max
# difference, the route is excluded from the PSL calculations. The route is still returned, but with a probability of
# 0.0.
#
# The `cutoff_prob` should be in the range [0, 1]. It is then rescaled internally to [0.5, 1] as probabilities below 0.5
# produce negative differences in utilities. A higher `cutoff_prob` includes more routes. A value of `0.0` will only
# include the minimum cost route. A value of `1.0` includes all routes.
#
# It is highly recommended to set either `max_routes` or `max_depth` to prevent runaway results.

# rc.set_choice_set_generation("link-penalisation", max_routes=5, penalty=1.02)
rc.set_choice_set_generation("bfsle", max_routes=5, beta=1.1)

# %%
# All parameters are optional, the defaults are:
print(rc.default_paramaters)

# %%
# We can now perform a computation for single OD pair if we'd like. Here we do one between the first and last centroid
# as well an an assignment.
results = rc.execute_single(77011, 74089, perform_assignment=True)
print(results[0])

# %%
# Because we asked it to also perform an assignment we can access the various results from that
# The default return is a Pyarrow Table but Pandas is nicer for viewing.
rc.get_results().to_pandas()

# %%
# To perform a batch operation we need to prepare the object first. We can either provide a list of tuple of the OD
# pairs we'd like to use, or we can provided a 1D list and the generation will be run on all permutations.
rc.prepare(graph.centroids[:5])  # You can inspect the result with rc.nodes

# %%
# Now we can perform a batch computation with an assignment
rc.execute(perform_assignment=True)
rc.get_results().to_pandas()

# %%
# Since we provided a matrix initially we can also perform link loading based on our assignment results.
rc.get_load_results()

# %%
# We can also plot these results
import folium
import geopandas as gpd

link_loads = rc.get_load_results()[0]
link_loads =link_loads[["link_id", "demand_tot"]]
max_load = link_loads["demand_tot"].max()

links = gpd.GeoDataFrame(project.network.links.data, crs=4326)
links = links.merge(link_loads, on="link_id")

links = links[links["demand_tot"] > 0]

# %%
loads_lyr = folium.FeatureGroup("link_loads")

# Maximum thickness we would like is probably a 10, so let's make sure we don't go over that
factor = 10 / max_load

# Let's create the layers
for _, rec in links.iterrows():
    points = rec.geometry.wkt.replace("LINESTRING ", "").replace("(", "").replace(")", "").split(", ")
    points = "[[" + "],[".join([p.replace(" ", ", ") for p in points]) + "]]"
    # we need to take from x/y to lat/long
    points = [[x[1], x[0]] for x in eval(points)]
    _ = folium.vector_layers.PolyLine(points, color="red", weight=factor * rec.demand_tot).add_to(loads_lyr)

# %%
# Create the map and center it in the correct place
long, lat = project.conn.execute("select avg(xmin), avg(ymin) from idx_links_geometry").fetchone()

map_osm = folium.Map(location=[lat, long], tiles="Cartodb Positron", zoom_start=12)
loads_lyr.add_to(map_osm)
folium.LayerControl().add_to(map_osm)
map_osm


# %%
# Select link analysis
# ~~~~~~~~~~~~~~~~~~
# We can also enable select link analysis by providing the links and the directions that we are interested in
rc.set_select_links({"sl1": [(5372, 1), (5374, 1)], "sl2": [(23845, 0)]})

# %%
# We can get then the results in a Pandas data frame for both the network and compressed graph.
u_sl, c_sl = rc.get_select_link_results()
u_sl

# %%
# We can also access the OD matrices for this link loading. These matrices are sparse and can be converted to
# scipy.sparse matrices for ease of use. They're stored in a dictionary where the key is the matrix name concatenated
# wit the select link set name via an underscore. These matrices are constructed during `get_select_link_results`.
list(rc.sl_od_matrix.keys())

# %%
od_matrix = rc.sl_od_matrix["demand_sl1"]
od_matrix.to_scipy()

# %%
project.close()
