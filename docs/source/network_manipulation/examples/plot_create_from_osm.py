"""
.. _plot_from_osm:

Create project from OpenStreetMap
=================================

In this example, we show how to create an empty project and populate it with a network from OpenStreetMap.

This time we will use GeoPandas to visualize the network.
"""
# %%
# .. admonition:: References
# 
#   * :ref:`importing_from_osm` 

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.project.Network.create_from_osm`

# %%

# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae import Project
import folium
# sphinx_gallery_thumbnail_path = '../source/_images/nauru.png'

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
# We can also choose to create a model from a polygon (which must be in ``EPSG:4326``)
# or from a Polygon defined by a bounding box, for example.

# project.network.create_from_osm(model_area=box(-112.185, 36.59, -112.179, 36.60))

# %%
# We grab all the links data as a geopandas GeoDataFrame so we can process it easier
links = project.network.links.data

# %%
# Let's plot our network!
map_osm = links.explore(color="blue", weight=10, tooltip="link_type", popup="link_id", name="links")
folium.LayerControl().add_to(map_osm)
map_osm

# %%
project.close()
