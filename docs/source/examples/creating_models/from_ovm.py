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
Project from Overture Maps
=============================
In this example, we show how to create an empty project and populate it with a network from Overture Maps.
We will use Folium to visualize the network.
"""

# %%
# Imports
from pathlib import Path
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae import Project
import folium

# %%
# We create an empty project on an arbitrary folder
from shutil import rmtree

fldr = join(gettempdir(), uuid4().hex)
project = Project()
project.new(fldr)
# %%
# Now we can download the network from any place in the world (as long as you have memory for all the download
# and data wrangling that will be done)
# We have stored Airlie Beach's transportation parquet files in the folder with the file path data_source below as using the cloud-native Parquet files takes a much longer time to run
# We recommend downloading these cloud-native Parquet files to drive and replacing the data_source file to match
dir = str(Path('../../../../').resolve())
data_source = Path(dir) / 'tests' / 'data' / 'overture' / 'theme=transportation'
output_dir = Path(fldr) / "theme=transportation"

# For the sake of this example, we will choose the small town of Airlie Beach.
# The "bbox" parameter specifies the bounding box encompassing the desired geographical location. In the given example, this refers to the bounding box that encompasses Airlie Beach.
bbox = [148.7077, -20.2780, 148.7324, -20.2621 ]
 
# We can create from a bounding box or a named place.
project.network.create_from_ovm(west=bbox[0], south=bbox[1], east=bbox[2], north=bbox[3], data_source=data_source, output_dir=data_source)

# %%
# We grab all the links data as a Pandas DataFrame so we can process it easier
links = project.network.links.data

# %%
# We create a Folium layer
network_links = folium.FeatureGroup("links")

# We do some Python magic to transform this dataset into the format required by Folium
# We are only getting link_id and link_type into the map, but we could get other pieces of info as well
for i, row in links.iterrows():
    points = row.geometry.wkt.replace("LINESTRING ", "").replace("(", "").replace(")", "").split(", ")
    points = "[[" + "],[".join([p.replace(" ", ", ") for p in points]) + "]]"
    # we need to take from x/y to lat/long
    points = [[x[1], x[0]] for x in eval(points)]

    line = folium.vector_layers.PolyLine(
        points, popup=f"<b>link_id: {row.link_id}</b>", tooltip=f"{row.link_type}", color="blue", weight=10
    ).add_to(network_links)

# %%
# We get the center of the region
long = (bbox[0]+bbox[2])/2
lat = (bbox[1]+bbox[3])/2

# %%
map_osm = folium.Map(location=[lat, long], zoom_start=14)
network_links.add_to(map_osm)
folium.LayerControl().add_to(map_osm)
map_osm

# %%
project.close()

# %%
