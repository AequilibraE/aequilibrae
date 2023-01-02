"""
Import GTFS
===========

On this example, we import a GTFS feed to our model. We will also perform map matching.

We use data from Coquimbo, a city in La Serena Metropolitan Area in Chile.

"""
# %%
## Imports
from math import sqrt
from uuid import uuid4
import os
from tempfile import gettempdir

from shapely import wkb
from shapely.geometry import Point

import urllib
from aequilibrae.transit import Transit
from aequilibrae.project import Project

"""Let's create an empty project on an arbitrary folder."""
# %%
fldr = os.path.join(gettempdir(), uuid4().hex)

project = Project()
project.new(fldr)

# For the sake of this example, we create the network from a bounding box.
# It should take about 1 minute to download all the network.
project.network.create_from_osm(west=-71.35, south=-30.07, east=-71.18, north=-29.87)

"""
Before we import the GTFS feed, let's create an arbitrary zoning system for our model.
We'll follow the same workflow as presented [here](https://www.aequilibrae.com/python/latest/_auto_examples/plot_create_zoning.html).
"""
# %%
zones = 350

network = project.network
geo = network.convex_hull()

zone_area = geo.area / zones
zone_side = sqrt(2 * sqrt(3) * zone_area / 9)

extent = network.extent()

curr = project.conn.cursor()
b = extent.bounds
curr.execute(
    "select st_asbinary(HexagonalGrid(GeomFromWKB(?), ?, 0, GeomFromWKB(?)))",
    [extent.wkb, zone_side, Point(b[2], b[3]).wkb],
)
grid = curr.fetchone()[0]
grid = wkb.loads(grid)

grid = [p for p in grid.geoms if p.intersects(geo)]

zoning = project.zoning
for i, zone_geo in enumerate(grid):
    zone = zoning.new(i + 1)
    zone.geometry = zone_geo
    zone.save()
    zone.add_centroid(None)

"""Let's download the GTFS feed."""
# %%
dest_path = os.path.join(fldr, "coquimbo.zip")
urllib.request.urlretrieve("http://datos.gob.cl/dataset/c77c9a50-6dd1-449d-b5ab-947ec0139b31/resource/a4edcf07-0657-456d-bbbc-54b2aec1de8d/download/coquimbo10feb16.zip", dest_path)

"""
Now we create our Transit object and import the GTFS feed into our model.
"""
# %%
data = Transit(project)
# Now we 
transit = data.new_gtfs(agency="LISERCO, LINCOSUR, LISANCO", file_path=dest_path)

"""
To load the data, we must chose one date. Our GTFS dates ranges between 2015-04-01 and 2019-12-31.
We're going to continue with 2016-04-13, but feel free to experiment any other available dates.
It should take approximately 2 minutes to load the data.
"""
# %%
transit.load_date("2016-04-13")

"""
Now we execute the map matching to find the real paths.
Depending on the GTFS size, this process can be really time consuming.
It should take about 8 minutes to map-match all routes and patterns.
"""
# %%
transit.set_allow_map_match(True)
transit.map_match()

"""Finally, we save our GTFS into our model."""
# %%
transit.save_to_disk()

# %%
project.close()
