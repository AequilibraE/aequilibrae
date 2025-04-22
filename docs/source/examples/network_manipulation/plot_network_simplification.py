"""
.. _plot_network_simplification:

Network simplification
======================

In this example we use Nauru network to show how one can simplify the network, 
merging short links into longer ones or turning links into nodes, and saving
theses changes into the project.

We use Folium to visualize the resulting network.
"""

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.project.tools.network_simplifcation`

# %%
# Imports

from uuid import uuid4
from tempfile import gettempdir
from os.path import join

from aequilibrae.utils.create_example import create_example
from aequilibrae.project.tools.network_simplifier import NetworkSimplifier

# %%

# Let's use the Nauru example project for display

fldr = join(gettempdir(), uuid4().hex)

project = create_example(fldr, "nauru")

# %%
# To simplify the network, we need to create a graph. As Nauru doesn't have any centroid in its network
# we have to create a centroid from an arbitrary node, otherwise we cannot create a graph.

nodes = project.network.nodes
centroid_count = nodes.data.query('is_centroid == 1').shape[0]

if centroid_count == 0:
    arbitrary_node = nodes.data["node_id"][0]
    nd = nodes.get(arbitrary_node)
    nd.is_centroid = 1
    nd.save()

# %%
    
# Let's analyze the mode car or 'c' in our model
mode = "c"

# Let's set the graph for computation
network = project.network
network.build_graphs(modes=[mode])
graph = network.graphs[mode]
graph.set_graph("distance")
graph.set_skimming("distance")
graph.set_blocked_centroid_flows(False)

# %%

# Let's revert to setting up that node as centroid in case we had to do it
if centroid_count == 0:
    nd.is_centroid = 0
    nd.save()

# %%
# We check the number of links and nodes our project has initially.

links_before = project.network.links.data
nodes_before = project.network.nodes.data

print("This project initially has {} links and {} nodes".format(links_before.shape[0], nodes_before.shape[0]))

# %%
# Let's call the `NetworkSimplifier` class
net = NetworkSimplifier()

# When we choose to simplify the network, we pass a graph object to the function,
# and the output of this operation is 

net.simplify(graph)
net.rebuild_network()

# %%
# Let's plot the previous and actual networks

# %%
# Differently we can simplify the network by collapsing links into nodes.
# Notice that this operation modifies the network in the neighborhood.

net.collapse_links_into_nodes([903])
net.rebuild_network()

# %%
# Let's plot the network once again and check the modifications.

# %%
project.close()