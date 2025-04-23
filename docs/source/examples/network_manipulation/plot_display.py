"""
.. _explore_network_on_notebook:

Exploring the network on a notebook
===================================

In this example, we show how to use Folium to plot a network for different modes.

We will need Folium for this example, and we will focus on creating a layer for
each mode in the network, a layer for all links and a layer for all nodes.
"""
# %%

# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join

from aequilibrae.utils.create_example import create_example
import folium
# sphinx_gallery_thumbnail_path = '../source/_images/plot_network_image.png'

# %%

# We create an empty project on an arbitrary folder
fldr = join(gettempdir(), uuid4().hex)

# Let's use the Nauru example project for display
project = create_example(fldr, "nauru")

# %%
# We grab all the links data as a geopandas GeoDataFrame so we can process it easier
links = project.network.links.data
nodes = project.network.nodes.data

# %%
# And if you want to take a quick look in your GeoDataFrames, you can plot it!

# links.plot()
 
# %%
# Let's create copies of our link layers for each mode
bike = links[links["modes"].str.contains("b")]
car = links[links["modes"].str.contains("c")]
transit = links[links["modes"].str.contains("t")]
walk = links[links["modes"].str.contains("w")]

# %%
# And plot out data!

map = links.explore(color="gray", style_kwds={"weight": 2}, popup="link_id", tooltip="modes", name="network_links")
map = nodes.explore(m=map, color="black", style_kwds={"radius": 5, "fillOpacity": 1.0}, name="network_nodes")

map = walk.explore(m=map, color="green", style_kwds={"weight": 3}, popup="link_id", tooltip="modes", name="walk")
map = bike.explore(m=map, color="green", style_kwds={"weight": 3}, popup="link_id", tooltip="modes", name="bike")
map = car.explore(m=map, color="red", style_kwds={"weight": 3}, popup="link_id", tooltip="modes", name="car")
map = transit.explore(m=map, color="yellow", style_kwds={"weight": 3}, popup="link_id", tooltip="modes", name="transit")

folium.LayerControl().add_to(map)
map

# %%
project.close()
