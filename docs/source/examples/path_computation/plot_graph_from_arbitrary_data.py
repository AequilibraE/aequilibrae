"""
.. _plot_graph_from_arbitrary_data:

Graph from arbitrary data
=========================

In this example, we demonstrate how to create an AequilibraE Graph from an arbitrary network.

We are using 
`Sioux Falls data <https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls>`_, 
from TNTP.
"""

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.paths.graph`

# %%

# Imports
import numpy as np
import pandas as pd

from aequilibrae.paths import Network
# sphinx_gallery_thumbnail_path = '../source/_images/graph_network.png'

# %%
# We start by adding the path to load our arbitrary network.
net_file = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/SiouxFalls/SiouxFalls_net.tntp"

# %%
# Let's read our data! We'll be using Sioux Falls transportation network data, but without
# geometric information. The data will be stored in a Pandas DataFrame containing information
# about initial and final nodes, link distances, travel times, etc.
net = pd.read_csv(net_file, skiprows=8, sep="\t", lineterminator="\n", usecols=np.arange(1, 11))

# %%
# The Network requires link_id, a_node, b_node, and direction.
# 
# We need to manipulate the data to add the missing fields (link_id and direction) and
# rename the node columns accordingly.
net.insert(0, "link_id", np.arange(1, net.shape[0] + 1))
net = net.assign(direction=1)
net.rename(columns={"init_node": "a_node", "term_node": "b_node"}, inplace=True)

# %%
# Now we can take a look in our network file
net.head()

# %%
# Create the source network, then prepare its routing arrays.
network = Network(net, centroids=np.arange(1, 25))
graph = network.build_graph(
    mode="c", cost="length", skimming=["length", "free_flow_time"], block_centroid_flows=False
)

# %%
# Two of AequilibraE's new features consist in directly computing path or skims.
#
# Let's compute the path between nodes 1 and 17...
res = graph.compute_path(1, 17)

# %%
# ... and print the corresponding nodes...
res.path_nodes

# %%
# ... and the path links.
res.path
# %%
# For path computation, when we call the method ``graph.compute_path(1, 17)``, we are calling the class
# ``PathComputation`` and storing its results into a variable.
# 
# Notice that other methods related to path computation, such as ``milepost`` can also be used with
# ``res``.

# %%
# For skim computation, the process is quite similar. When calligng the method ``graph.compute_skims()``
# we are actually calling the class ``NetworkSkimming``, and storing its results into ``skm``.

skm = graph.compute_skims()

# %%
# Let's get the values for 'free_flow_time' matrix.
skims = skm.results.skims
skims["free_flow_time"]

# %%
# Now we're all set!

# %%
# Graph image credits to
# `Behance-network icons created by Sumitsaengtong - Flaticon <https://www.flaticon.com/free-icons/behance-network>`_