"""
.. _example_gtfs:

Import GTFS
===========

In this example, we import a GTFS feed to our model and perform map matching. 

We use data from Coquimbo, a city in La Serena Metropolitan Area in Chile.
"""
# %%

# Imports
from uuid import uuid4
from os import remove
from os.path import join
from tempfile import gettempdir

import folium
import pandas as pd
from aequilibrae.project.database_connection import database_connection

from aequilibrae.transit import Transit
from aequilibrae.utils.create_example import create_example
# sphinx_gallery_thumbnail_path = 'images/plot_import_gtfs.png'

# %%
# Let's create an empty project on an arbitrary folder.
fldr = join(gettempdir(), uuid4().hex)
project = create_example(fldr, "coquimbo")

# %%

# Since the Coquimbo example already includes a complete GTFS model, we will remove its public transport 
# database for the purposes of this example.    
remove(join(fldr, "public_transport.sqlite"))

# %%
# Let's import the GTFS feed.
dest_path = join(fldr, "gtfs_coquimbo.zip")

# %%
# Now we create our Transit object. This will automatically create a new public transport database.
data = Transit(project)

# %%
# To initialize the GTFS builder, specify the path to the GTFS file. You no longer need to provide the 
# name of the transit agency, but you can add a general description of the GTFS feed. If your GTFS file
# includes multiple transit agencies, all data will be loaded simultaneously. However, any description 
# you add will apply to all agencies in the file.
transit = data.new_gtfs_builder(file_path=dest_path, description="Wednesday feed by John Doe")

#%%
# Case you want information on the available dates before loading the GTFS data to the database,
# it is possible to use the function ``transit.dates_available()`` to check the available feed dates.

# %%
# To load the data, we must choose one date using the format ``YYYY-MM-DD``. We're going to build
# our database using the day 2016-04-13 but feel free to experiment with any other available dates.
# 
# It shouldn't take long to load the data.

transit.load_date("2016-04-13")

# %%
# Now we execute the map matching to find the real paths. Depending on the number or different route  
# patterns and/or the project area size, this process can be really time-consuming.
transit.map_match()

# %%
# Finally, we save our GTFS feed into our model.
transit.save_to_disk()

# %%
# Now we will plot the route's patterns we just imported
conn = database_connection("transit")

links = pd.read_sql("SELECT pattern_id, ST_AsText(geometry) geom FROM routes;", con=conn)
stops = pd.read_sql("SELECT stop_id, ST_X(geometry) X, ST_Y(geometry) Y FROM stops;", con=conn)

# %%
gtfs_links = folium.FeatureGroup("links")
gtfs_stops = folium.FeatureGroup("stops")

layers = [gtfs_links, gtfs_stops]

# %%
pattern_colors = ["#146DB3", "#EB9719"]

# %%
for i, row in links.iterrows():
    points = row.geom.replace("MULTILINESTRING", "").replace("(", "").replace(")", "").split(", ")
    points = "[[" + "],[".join([p.replace(" ", ", ") for p in points]) + "]]"
    points = [[x[1], x[0]] for x in eval(points)]

    _ = folium.vector_layers.PolyLine(
        points, 
        popup=f"<b>pattern_id: {row.pattern_id}</b>", 
        color=pattern_colors[i], 
        weight=5,
    ).add_to(gtfs_links)

for i, row in stops.iterrows():
    point = (row.Y, row.X)

    _ = folium.vector_layers.CircleMarker(
        point,
        popup=f"<b>stop_id: {row.stop_id}</b>",
        color="black",
        radius=2,
        fill=True,
        fillColor="black",
        fillOpacity=1.0,
    ).add_to(gtfs_stops)

#%%
# Let's create the map!

# %%
map_osm = folium.Map(location=[-29.93, -71.29], zoom_start=13)

# add all layers
for layer in layers:
    layer.add_to(map_osm)

# And add layer control before we display it
folium.LayerControl().add_to(map_osm)
map_osm

# %%
project.close()
