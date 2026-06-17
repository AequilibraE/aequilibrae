"""
.. _plot_from_osm:

Create project from OpenStreetMap
=================================

In this example we show how to create an empty project and populate it with a
network from OpenStreetMap. The new pluggable network-acquisition framework
(``Network.import_from_osm``) replaces the old ``create_from_osm`` API.

We use GeoPandas to visualise the result.

Install the optional dependencies first::

    pip install aequilibrae[create]
"""
# %%
# .. admonition:: References
#
#   * :ref:`importing_from_osm`

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.project.network.network.Network.import_from_osm`

# %%

# Imports
from os.path import join
from tempfile import gettempdir
from uuid import uuid4

from aequilibrae import Project
# sphinx_gallery_thumbnail_path = '../source/_images/nauru.png'

# %%

# Create an empty project on an arbitrary folder
fldr = join(gettempdir(), uuid4().hex)

project = Project()
project.new(fldr)

# %%
# Import a network for the small nation of Nauru. The raw Overpass response is
# saved to ``<project>/downloaded data/osm-overpass/`` so the import can be
# inspected or replayed offline later.
project.network.import_from_osm(place_name="Nauru")

# %%
# We can also import from a polygon (which must be in EPSG:4326) or from a
# bounding box, or from a local .osm.pbf file:
#
# .. code-block:: python
#
#     from shapely.geometry import box
#     project.network.import_from_osm(
#         model_area=box(-112.185, 36.59, -112.179, 36.60)
#     )
#     # or
#     project.network.import_from_osm(pbf_path="path/to/extract.osm.pbf")

# %%
# Grab all the links as a GeoDataFrame and plot.
links = project.network.links.data
links.explore(color="blue", style_kwds={"weight": 2}, tooltip="link_type")

# %%
project.close()
