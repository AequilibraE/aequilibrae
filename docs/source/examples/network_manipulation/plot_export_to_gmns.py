"""
.. _export_to_gmns:

Exporting network to GMNS
===========================

In this example, we export a simple network to GMNS format.
The source AequilibraE model used as input for this is the result of the import process
(``create_from_gmns()``) using the GMNS example of Arlington Signals, which can be found
in the GMNS repository on GitHub: https://github.com/zephyr-data-specs/GMNS
"""
# %%
# .. admonition:: References
# 
#   * :ref:`aequilibrae_to_gmns` 

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.project.network.network.Network.export_to_gmns`

# %%

# Imports
from uuid import uuid4
import os
from tempfile import gettempdir

from aequilibrae.utils.create_example import create_example
import folium
import geopandas as gpd
import pandas as pd
# sphinx_gallery_thumbnail_path = '../source/_images/plot_export_to_gmns.png'

# %%

# We load the example project inside a temp folder
fldr = os.path.join(gettempdir(), uuid4().hex)

project = create_example(fldr)

# %%
# We export the network to CSV files in GMNS format, that will be saved inside the project folder
output_fldr = os.path.join(gettempdir(), uuid4().hex)
if not os.path.exists(output_fldr):
    os.mkdir(output_fldr)

project.network.export_to_gmns(path=output_fldr)

# %%
# Now, let's plot a map. This map can be compared with the images of the README.md
# file located in this example repository on GitHub:
# https://github.com/zephyr-data-specs/GMNS/blob/develop/examples/Arlington_Signals/README.md
links = pd.read_csv(os.path.join(output_fldr, "link.csv"))
nodes = pd.read_csv(os.path.join(output_fldr, "node.csv"))

# %%
# We turn the links and nodes DataFrames into GeoDataFrames so we can plot them more easily.
links = gpd.GeoDataFrame(links, geometry=gpd.GeoSeries.from_wkt(links["geometry"]), crs=4326)
nodes = gpd.GeoDataFrame(nodes, geometry=gpd.GeoSeries.from_xy(nodes["x_coord"], nodes["y_coord"]), crs=4326)

# %%
# Let's plot our map!

map = links.explore(color="black", style_kwds={"weight": 2}, tool_tip="link_type", name="links")
map = nodes.explore(m=map, color="red", style_kwds={"radius": 5, "fillOpacity": 1.0}, name="nodes")

folium.LayerControl().add_to(map) # Add a layer control button to our map
map
# %%
project.close()
