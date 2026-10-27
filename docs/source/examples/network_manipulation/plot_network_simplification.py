"""
.. _plot_network_simplifier:

Network simplifier
==================

In this example we use Nauru network to show how one can simplify the network,
merging short links into longer ones or turning links into nodes, and saving
these changes into the project.

We use Folium to visualize the resulting network.
"""

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.project.tools.network_simplifier`

# %%

# Imports
from os.path import join
from tempfile import gettempdir
from uuid import uuid4

import branca
import folium

from aequilibrae.project.tools.network_simplifier import NetworkSimplifier
from aequilibrae.utils.create_example import create_example

# sphinx_gallery_thumbnail_path = '../source/_images/plot_net_simplifier.png'

# %%
# Let's use the Nauru example project for display

fldr = join(gettempdir(), uuid4().hex)

project = create_example(fldr, "nauru")

# %%
# Let's analyze the mode car or 'c' in our model
mode = "c"

# %%
# We check the number of links and nodes our project has initially.

links_before = project.network.links.data
nodes_before = project.network.nodes.data

print("This project initially has {} links and {} nodes".format(links_before.shape[0], nodes_before.shape[0]))

# %%
# Let's call the ``NetworkSimplifier`` class. Any changes made to the database using this class
# are permanent. Make sure you have a backup if necessary.
net = NetworkSimplifier(project)

# %%
# Simplification prepares its routing data from the project network.

net.simplify(mode)

# %%
# Let's plot the previous and actual networks!

links_after = net.network.links.data
nodes_after = net.network.nodes.data

# %%
fig = branca.element.Figure()

subplot1 = fig.add_subplot(1, 2, 1)
subplot2 = fig.add_subplot(1, 2, 2)

map1 = folium.Map(location=[-0.508371, 166.931142], zoom_start=17)
map1 = links_before.explore(m=map1, color="black", style_kwds={"weight": 2}, name="links_before")
map1 = nodes_before.explore(m=map1, color="red", style_kwds={"radius": 3, "fillOpacity": 1.0}, name="nodes_before")
folium.LayerControl().add_to(map1)

map2 = folium.Map(location=[-0.508371, 166.931142], zoom_start=17)
map2 = links_after.explore(m=map2, color="black", style_kwds={"weight": 2}, name="links_after")
map2 = nodes_after.explore(m=map2, color="blue", style_kwds={"radius": 3, "fillOpacity": 1.0}, name="nodes_after")
folium.LayerControl().add_to(map2)

subplot1.add_child(map1)
subplot2.add_child(map2)

fig
# %%
# Differently we can simplify the network by collapsing links into nodes.
# Notice that this operation modifies the network in the neighborhood.

net.collapse_links_into_nodes([903])

# %%
# Let's plot the network once again and check the modifications!

links_after = net.network.links.data
nodes_after = net.network.nodes.data

# %%
fig = branca.element.Figure()

subplot1 = fig.add_subplot(1, 2, 1)
subplot2 = fig.add_subplot(1, 2, 2)

map1 = folium.Map(location=[-0.509363, 166.928563], zoom_start=18)
map1 = links_before.explore(m=map1, color="black", style_kwds={"weight": 2}, name="links_before")
map1 = nodes_before.explore(m=map1, color="red", style_kwds={"radius": 3, "fillOpacity": 1.0}, name="nodes_before")
folium.LayerControl().add_to(map1)

map2 = folium.Map(location=[-0.509363, 166.928563], zoom_start=18)
map2 = links_after.explore(m=map2, color="black", style_kwds={"weight": 2}, name="links_after")
map2 = nodes_after.explore(m=map2, color="blue", style_kwds={"radius": 3, "fillOpacity": 1.0}, name="nodes_after")
folium.LayerControl().add_to(map2)

subplot1.add_child(map1)
subplot2.add_child(map2)

fig
# %%
project.close()
