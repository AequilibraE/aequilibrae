"""
.. _plot_assignment_without_model:

Traffic Assignment without an AequilibraE Model
===============================================

In this example, we show how to perform Traffic Assignment in AequilibraE without a model.

We are using `Sioux Falls data <https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls>`_, from TNTP.
"""
# %%
# .. admonition:: References
# 
#   * :ref:`static_traffic_assignment`

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.paths.Graph`
#     * :func:`aequilibrae.paths.TrafficClass`
#     * :func:`aequilibrae.paths.TrafficAssignment` 
#     * :func:`aequilibrae.matrix.AequilibraeMatrix`

# %%

# Imports
import os
import pandas as pd
import numpy as np
from uuid import uuid4
from tempfile import gettempdir

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph
from aequilibrae.paths import TrafficAssignment
from aequilibrae.paths.traffic_class import TrafficClass
# sphinx_gallery_thumbnail_path = '../source/images/assignment_plot.png'

# %%
# We load the example file from the GMNS GitHub repository
net_file = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/SiouxFalls/SiouxFalls_net.tntp"

demand_file = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/SiouxFalls/CSV-data/SiouxFalls_od.csv"

geometry_file = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/SiouxFalls/SiouxFalls_node.tntp"

# %%
# Let's use a temporary folder to store our data
folder = os.path.join(gettempdir(), uuid4().hex)

# %% 
# First we load our demand file. This file has three columns: O, D, and Ton. 
# O and D stand for origin and destination, respectively, and Ton is the demand of each
# OD pair.
dem = pd.read_csv(demand_file)
zones = int(max(dem.O.max(), dem.D.max()))
index = np.arange(zones) + 1

# %%
# Since our OD-matrix is in a different shape than we expect (for Sioux Falls, that
# would be a 24x24 matrix), we must create our matrix.
mtx = np.zeros(shape=(zones, zones))
for element in dem.to_records(index=False):
    mtx[element[0]-1][element[1]-1] = element[2]

# %%
# Now let's create an AequilibraE Matrix with out data
aemfile = os.path.join(folder, "demand.aem")
aem = AequilibraeMatrix()
kwargs = {'file_name': aemfile,
          'zones': zones,
          'matrix_names': ['matrix']}

aem.create_empty(**kwargs)
aem.matrix['matrix'][:,:] = mtx[:,:]
aem.index[:] = index[:]

# %%
# Let's import information about our network. As we're loading data in TNTP format,
# we should do these manipulations.
net = pd.read_csv(net_file, skiprows=2, sep="\t", lineterminator=";", header=None)

net.columns = ["newline", "a_node", "b_node", "capacity", "length", "free_flow_time", "b", "power", "speed", "toll", "link_type", "terminator"]

net.drop(columns=["newline", "terminator"], index=[76], inplace=True)

# %%
network = net[['a_node', 'b_node', "capacity", 'free_flow_time', "b", "power"]]
network = network.assign(direction=1)
network["link_id"] = network.index + 1
network = network.astype({"a_node":"int64", "b_node": "int64"})

# %%
# Now we'll import the geometry (as lon/lat) for our network, this is required if you plan to 
# use the `A*` path finding, otherwise it can safely be skipped.
geom = pd.read_csv(geometry_file, skiprows=1, sep="\t", lineterminator=";", header=None)
geom.columns = ["newline", "lon", "lat", "terminator"]
geom.drop(columns=["newline", "terminator"], index=[24], inplace=True)
geom["node_id"] = geom.index + 1
geom = geom.astype({"node_id": "int64", "lon": "float64", "lat": "float64"}).set_index("node_id")

# %%
# Let's build our Graph! In case you're in doubt about AequilibraE Graph, 
# :ref:`click here <aequilibrae-graphs>` to read more about it.

# %%
g = Graph()
g.cost = network['free_flow_time'].values
g.capacity = network['capacity'].values
g.free_flow_time = network['free_flow_time'].values

g.network = network
g.prepare_graph(index)
g.set_graph("free_flow_time")
g.cost = np.array(g.cost, copy=True)
g.set_skimming(["free_flow_time"])
g.set_blocked_centroid_flows(False)
g.network["id"] = g.network.link_id
g.lonlat_index = geom.loc[g.all_nodes]

# %%
# Let's prepare our matrix for computation
aem.computational_view(["matrix"])

# %%
# Let's perform our assignment. Feel free to try different algorithms,
# as well as change the maximum number of iterations and the gap
assigclass = TrafficClass("car", g, aem)

assig = TrafficAssignment()

assig.set_classes([assigclass])
assig.set_vdf("BPR")
assig.set_vdf_parameters({"alpha": "b", "beta": "power"})
assig.set_capacity_field("capacity")
assig.set_time_field("free_flow_time")
assig.set_algorithm("fw")
assig.max_iter = 100
assig.rgap_target = 1e-6
assig.execute()

# %%
# Now let's take a look at the Assignment results
assig.results()

# %%
# And at the Assignment report
assig.report()
