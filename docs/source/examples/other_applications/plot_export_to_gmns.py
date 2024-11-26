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
#     * :func:`aequilibrae.project.Network.export_to_gmns`

# %%

# Imports
from uuid import uuid4
import os
from tempfile import gettempdir
from aequilibrae.utils.create_example import create_example
import pandas as pd
import folium
# sphinx_gallery_thumbnail_path = '../source/images/plot_export_to_gmns.png'

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

# We create our Folium layers
network_links = folium.FeatureGroup("links")
network_nodes = folium.FeatureGroup("nodes")
layers = [network_links, network_nodes]

# We do some Python magic to transform this dataset into the format required by Folium
# We are only getting link_id and link_type into the map, but we could get other pieces of info as well
for i, row in links.iterrows():
    points = row.geometry.replace("LINESTRING ", "").replace("(", "").replace(")", "").split(", ")
    points = "[[" + "],[".join([p.replace(" ", ", ") for p in points]) + "]]"
    # we need to take from x/y to lat/long
    points = [[x[1], x[0]] for x in eval(points)]

    _ = folium.vector_layers.PolyLine(
        points, popup=f"<b>link_id: {row.link_id}</b>", tooltip=f"{row.facility_type}", color="black", weight=2
    ).add_to(network_links)

# And now we get the nodes
for i, row in nodes.iterrows():
    point = (row.y_coord, row.x_coord)

    _ = folium.vector_layers.CircleMarker(
        point,
        popup=f"<b>link_id: {row.node_id}</b>",
        tooltip=f"{row.node_type}",
        color="red",
        radius=5,
        fill=True,
        fillColor="red",
        fillOpacity=1.0,
    ).add_to(network_nodes)

# %%

# We get the center of the region
curr = project.conn.cursor()
curr.execute("select avg(xmin), avg(ymin) from idx_links_geometry")
long, lat = curr.fetchone()

# We create the map
map_gmns = folium.Map(location=[lat, long], zoom_start=12)

# add all layers
for layer in layers:
    layer.add_to(map_gmns)

# And Add layer control before we display it
folium.LayerControl().add_to(map_gmns)
map_gmns

# %%
project.close()
