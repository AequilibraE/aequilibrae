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
# sphinx_gallery_thumbnail_path = '../source/images/plot_network_image.png'

# %%

# We create an empty project on an arbitrary folder
fldr = join(gettempdir(), uuid4().hex)

# %%
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
# We create our Folium layers
network_links = folium.FeatureGroup("links")
network_nodes = folium.FeatureGroup("nodes")
car = folium.FeatureGroup("Car")
walk = folium.FeatureGroup("Walk")
bike = folium.FeatureGroup("Bike")
transit = folium.FeatureGroup("Transit")
layers = [network_links, network_nodes, car, walk, bike, transit]

# %%
# We do some Python magic to transform this dataset into the format required by Folium
# We are only getting link_id and link_type into the map, but we could get other pieces of info as well
for i, row in links.iterrows():
    points = row.geometry.wkt.replace("LINESTRING ", "").replace("(", "").replace(")", "").split(", ")
    points = "[[" + "],[".join([p.replace(" ", ", ") for p in points]) + "]]"
    # we need to take from x/y to lat/long
    points = [[x[1], x[0]] for x in eval(points)]

    _ = folium.vector_layers.PolyLine(
        points, popup=f"<b>link_id: {row.link_id}</b>", tooltip=f"{row.modes}", color="gray", weight=2
    ).add_to(network_links)

    if "w" in row.modes:
        _ = folium.vector_layers.PolyLine(
            points, popup=f"<b>link_id: {row.link_id}</b>", tooltip=f"{row.modes}", color="green", weight=4
        ).add_to(walk)

    if "b" in row.modes:
        _ = folium.vector_layers.PolyLine(
            points, popup=f"<b>link_id: {row.link_id}</b>", tooltip=f"{row.modes}", color="green", weight=4
        ).add_to(bike)

    if "c" in row.modes:
        _ = folium.vector_layers.PolyLine(
            points, popup=f"<b>link_id: {row.link_id}</b>", tooltip=f"{row.modes}", color="red", weight=4
        ).add_to(car)

    if "t" in row.modes:
        _ = folium.vector_layers.PolyLine(
            points, popup=f"<b>link_id: {row.link_id}</b>", tooltip=f"{row.modes}", color="yellow", weight=4
        ).add_to(transit)

# And now we get the nodes
for i, row in nodes.iterrows():
    point = (row.geometry.y, row.geometry.x)

    _ = folium.vector_layers.CircleMarker(
        point,
        popup=f"<b>link_id: {row.node_id}</b>",
        tooltip=f"{row.modes}",
        color="black",
        radius=5,
        fill=True,
        fillColor="black",
        fillOpacity=1.0,
    ).add_to(network_nodes)

# %%
# We get the center of the region we are working with some SQL magic
curr = project.conn.cursor()
curr.execute("select avg(xmin), avg(ymin) from idx_links_geometry")
long, lat = curr.fetchone()

# %%
# We create the map
map_osm = folium.Map(location=[lat, long], zoom_start=14)

# add all layers
for layer in layers:
    layer.add_to(map_osm)

# And Add layer control before we display it
folium.LayerControl().add_to(map_osm)
map_osm

# %%
project.close()
