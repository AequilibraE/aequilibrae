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
import geopandas as gpd
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

patterns = pd.read_sql("SELECT pattern_id, ST_AsText(geometry) geom FROM routes;", con=conn)
stops = pd.read_sql("""SELECT stop_id, ST_X(geometry) X, ST_Y(geometry) Y FROM stops""", con=conn)

# %%
# We turn the patterns and stops DataFrames into GeoDataFrames so we can plot them more easily.
patterns = gpd.GeoDataFrame(patterns, geometry=gpd.GeoSeries.from_wkt(patterns["geom"]), crs=4326)
stops = gpd.GeoDataFrame(stops, geometry=gpd.GeoSeries.from_xy(stops["X"], stops["Y"]), crs=4326)

# %%
# And plot out data!

map = patterns.explore(color=["#146DB3", "#EB9719"], style_kwds={"weight": 4}, name="links")
map = stops.explore(m=map, color="black", style_kwds={"radius": 2, "fillOpacity": 1.0}, name="stops")

folium.LayerControl().add_to(map)
map

# %%
project.close()
