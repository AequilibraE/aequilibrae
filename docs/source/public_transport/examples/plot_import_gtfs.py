"""
.. _example_gtfs:

Import GTFS
===========

In this example, we import a GTFS feed to our model and perform map matching. 

We use data from Coquimbo, a city in La Serena Metropolitan Area in Chile.
"""
# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.transit.Transit`
#     * :func:`aequilibrae.transit.lib_gtfs.GTFSRouteSystemBuilder`

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

# sphinx_gallery_thumbnail_path = '../source/_images/plot_import_gtfs.png'

# %%

# Let's create an empty project on an arbitrary folder.
fldr = join(gettempdir(), uuid4().hex)
project = create_example(fldr, "coquimbo")

# %%
# As the Coquimbo example already has a complete GTFS model, we shall remove its public transport
# database for the sake of this example.
remove(join(fldr, "public_transport.sqlite"))

# %%
# Let's import the GTFS feed.
dest_path = join(fldr, "gtfs_coquimbo.zip")

# %%
# Now we create our Transit object and import the GTFS feed into our model.
# This will automatically create a new public transport database.

data = Transit(project)

transit = data.new_gtfs_builder(agency="Lisanco", file_path=dest_path)

# %%
# To load the data, we must choose one date. We're going to continue with 2016-04-13 but feel free
# to experiment with any other available dates. Transit class has a function allowing you to check
# dates for the GTFS feed. It should take approximately 2 minutes to load the data.

transit.load_date("2016-04-13")

# Now we execute the map matching to find the real paths.
# Depending on the GTFS size, this process can be really time-consuming.

# transit.set_allow_map_match(True)
# transit.map_match()

# Finally, we save our GTFS into our model.
transit.save_to_disk()

# %%
# Now we will plot one of the route's patterns we just imported
conn = database_connection("transit")

links = pd.read_sql("SELECT pattern_id, ST_AsText(geometry) geom FROM routes;", con=conn)

stops = pd.read_sql("""SELECT stop_id, ST_X(geometry) X, ST_Y(geometry) Y FROM stops""", con=conn)

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

# %%
# Let's create the map!
map_osm = folium.Map(location=[-29.93, -71.29], zoom_start=13)

# add all layers
for layer in layers:
    layer.add_to(map_osm)

# And add layer control before we display it
folium.LayerControl().add_to(map_osm)
map_osm

# %%
project.close()
