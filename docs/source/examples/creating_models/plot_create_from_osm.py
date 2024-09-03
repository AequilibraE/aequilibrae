"""
.. _plot_from_osm:

Create project from OpenStreetMap
=================================

In this example, we show how to create an empty project and populate it with a network from OpenStreetMap.

This time we will use Folium to visualize the network.
"""

# %%

# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae import Project
import folium
# sphinx_gallery_thumbnail_path = 'images/nauru.png'

# %%
# We create an empty project on an arbitrary folder
fldr = join(gettempdir(), uuid4().hex)
project = Project()
project.new(fldr)

# %%
# Now we can download the network from any place in the world (as long as you have memory for all the download
# and data wrangling that will be done).

# %%
# We can create from a bounding box or a named place.
# For the sake of this example, we will choose the small nation of Nauru.
project.network.create_from_osm(place_name="Nauru")

# %%
# We can also choose to create a model from a polygon (which must be in `EPSG:4326`)
# Or from a Polygon defined by a bounding box, for example.

# project.network.create_from_osm(model_area=box(-112.185, 36.59, -112.179, 36.60))

# %%
# We grab all the links data as a Pandas DataFrame so we can process it easier
links = project.network.links.data

# %%
# We create a Folium layer
network_links = folium.FeatureGroup("links")

# %%
# We do some Python magic to transform this dataset into the format required by Folium.
# We are only getting link_id and link_type into the map, but we could get other pieces of info as well.
for i, row in links.iterrows():
    points = row.geometry.wkt.replace("LINESTRING ", "").replace("(", "").replace(")", "").split(", ")
    points = "[[" + "],[".join([p.replace(" ", ", ") for p in points]) + "]]"
    # we need to take from x/y to lat/long
    points = [[x[1], x[0]] for x in eval(points)]

    line = folium.vector_layers.PolyLine(
        points, popup=f"<b>link_id: {row.link_id}</b>", tooltip=f"{row.link_type}", color="blue", weight=10
    ).add_to(network_links)

# %%
# We get the center of the region we are working with some SQL magic
curr = project.conn.cursor()
curr.execute("select avg(xmin), avg(ymin) from idx_links_geometry")
long, lat = curr.fetchone()

# %%
map_osm = folium.Map(location=[lat, long], zoom_start=14)
network_links.add_to(map_osm)
folium.LayerControl().add_to(map_osm)
map_osm

# %%
project.close()

# %%
# .. seealso::
#     The use of the following functions, methods, classes and modules is shown in this example:
#
#     * :func:`aequilibrae.project.Network.create_from_osm`
#     * :ref:`importing_from_osm`
