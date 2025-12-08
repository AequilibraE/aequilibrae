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
#     * :func:`aequilibrae.project.network.network.Network.create_from_gmns`

# %%

# Imports
from uuid import uuid4
from os.path import join
from tempfile import gettempdir

from aequilibrae import Project
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
map = links.explore(color="black", style_kwds={"weight": 2}, tool_tip="link_type", name="links")
map = nodes.explore(m=map, color="red", style_kwds={"radius": 5, "fillOpacity": 1.0}, name="nodes")

folium.LayerControl().add_to(map) # Add a layer control button to our map
map

# %%
project.close()
