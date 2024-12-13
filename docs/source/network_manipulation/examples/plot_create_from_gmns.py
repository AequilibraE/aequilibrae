# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %%
"""
.. _import_from_gmns:

Create project from GMNS
========================

In this example, we import a simple network in GMNS format.
The source files of this network are publicly available in the 
`GMNS GitHub repository <https://github.com/zephyr-data-specs/GMNS>`_ itself.
"""
# %%
# .. admonition:: References
# 
#   * :ref:`importing_from_gmns_file` 

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.project.Network.create_from_gmns`

# %%

# Imports
from uuid import uuid4
from os.path import join
from tempfile import gettempdir
from aequilibrae.project import Project
from aequilibrae.parameters import Parameters
import folium
# sphinx_gallery_thumbnail_path = '../source/_images/plot_import_from_gmns.png'

# %%

# We load the example file from the GMNS GitHub repository
link_file = "https://raw.githubusercontent.com/zephyr-data-specs/GMNS/main/examples/Arlington_Signals/link.csv"
node_file = "https://raw.githubusercontent.com/zephyr-data-specs/GMNS/main/examples/Arlington_Signals/node.csv"
use_group_file = "https://raw.githubusercontent.com/zephyr-data-specs/GMNS/main/examples/Arlington_Signals/use_group.csv"

# %%

# We create the example project inside our temp folder
fldr = join(gettempdir(), uuid4().hex)

project = Project()
project.new(fldr)

# %%
# In this cell, we modify the AequilibraE parameters.yml file so it contains additional
# fields to be read in the GMNS link and/or node tables. Remember to always keep the
# "required" key set to False, since we are adding a non-required field.
new_link_fields = {
    "bridge": {"description": "bridge flag", "type": "text", "required": False},
    "tunnel": {"description": "tunnel flag", "type": "text", "required": False},
}
new_node_fields = {
    "port": {"description": "port flag", "type": "text", "required": False},
    "hospital": {"description": "hospital flag", "type": "text", "required": False},
}

par = Parameters()
par.parameters["network"]["gmns"]["link"]["fields"].update(new_link_fields)
par.parameters["network"]["gmns"]["node"]["fields"].update(new_node_fields)
par.write_back()

# %%
# As it is specified that the geometries are in the coordinate system EPSG:32619,
# which is different than the system supported by AequilibraE (EPSG:4326), we inform
# the srid in the method call:
project.network.create_from_gmns(
    link_file_path=link_file, node_file_path=node_file, use_group_path=use_group_file, srid=32619
)

# %%
# Now, let's plot a map. This map can be compared with the images of the README.md
# file located in this example repository on GitHub:
# https://github.com/zephyr-data-specs/GMNS/blob/develop/examples/Arlington_Signals/README.md
links = project.network.links.data
nodes = project.network.nodes.data

# %%
# We create our Folium layers
network_links = folium.FeatureGroup("links")
network_nodes = folium.FeatureGroup("nodes")
layers = [network_links, network_nodes]

# %%
# We do some Python magic to transform this dataset into the format required by Folium
# We are only getting link_id and link_type into the map, but we could get other pieces of info as well
for i, row in links.iterrows():
    points = row.geometry.wkt.replace("LINESTRING ", "").replace("(", "").replace(")", "").split(", ")
    points = "[[" + "],[".join([p.replace(" ", ", ") for p in points]) + "]]"
    # we need to take from x/y to lat/long
    points = [[x[1], x[0]] for x in eval(points)]

    _ = folium.vector_layers.PolyLine(
        points, popup=f"<b>link_id: {row.link_id}</b>", tooltip=f"{row.modes}", color="black", weight=2
    ).add_to(network_links)

# %%
# And now we get the nodes

for i, row in nodes.iterrows():
    point = (row.geometry.y, row.geometry.x)

    _ = folium.vector_layers.CircleMarker(
        point,
        popup=f"<b>link_id: {row.node_id}</b>",
        tooltip=f"{row.modes}",
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

# %%

# We create the map
map_gmns = folium.Map(location=[lat, long], zoom_start=17)

# Add all layers
for layer in layers:
    layer.add_to(map_gmns)

# And Add layer control before we display it
folium.LayerControl().add_to(map_gmns)
map_gmns

# %%
project.close()
