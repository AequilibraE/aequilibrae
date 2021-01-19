"""
Creating a zone system based on Hex Bins
========================================

On this example we show how to create a hex bin zones covering an arbitrary area.

We use the Nauru example to create roughly 100 zones covering the whole modelling
area as delimited by the entire network

You are obviously welcome to create whatever zone system you would like, as long as
you have the geometries for them. In that case, you can just skip the Hex bin computation
part of this notebook.

We also add centroid connectors to our network to make it a pretty complete example
"""

# %%
# **What we want to create a zoning system like this**

# %%
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("plot_create_zoning.png")
plt.imshow(img)

# %%
## Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from math import sqrt
from shapely.geometry import Point
import shapely.wkb
from aequilibrae.utils.create_example import create_example

# %%
# We create an empty project on an arbitrary folder
fldr = join(gettempdir(), uuid4().hex)

# Let's use the Nauru example project for display
project = create_example(fldr, "nauru")

# %%
# We said we wanted 100 zones
zones = 100

# %% md
# Hex Bins using SpatiaLite


# %% md
#### Spatialite requires a few things to compute hex bins

# %%
# One of the them is the area you want to cover
network = project.network

# So we use the convenient network method convex_hull() (it may take some time for very large networks)
geo = network.convex_hull()

# %%
# The second thing is the side of the hexbin, which we can compute from its area
# The approximate area of the desired hexbin is
zone_area = geo.area / zones
# Since the area of the hexagon is **3 * sqrt(3) * side^2 / 2**
# is side is equal to  **sqrt(2 * sqrt(3) * A/9)**
zone_side = sqrt(2 * sqrt(3) * zone_area / 9)

# %%
# Now we can run an sql query to compute the hexagonal grid
# There are many ways to create Hex bins (including with a GUI on QGIS), but we find that
# using SpatiaLite is a pretty neat solution
# For which we will use the entire network bounding box to make sure we cover everything
extent = network.extent()

curr = project.conn.cursor()
b = extent.bounds
curr.execute(
    "select st_asbinary(HexagonalGrid(GeomFromWKB(?), ?, 0, GeomFromWKB(?)))",
    [extent.wkb, zone_side, Point(b[2], b[3]).wkb],
)
grid = curr.fetchone()[0]
grid = shapely.wkb.loads(grid)

# %%
# Since we used the bounding box, we have WAY more zones than we wanted, so we clean them
# by only keeping those that intersect the network convex hull.
grid = [p for p in grid if p.intersects(geo)]

# Let's re-number all nodes with IDs smaller than 300 to something bigger as to free space to our centroids to go from 1
# to N
nodes = network.nodes
for i in range(1, 301):
    nd = nodes.get(i)
    nd.renumber(i + 1300)

# %%
# Now we can add them to the model
# And add centroids to them while we are at it
zoning = project.zoning
for i, zone_geo in enumerate(grid):
    zone = zoning.new(i + 1)
    zone.geometry = zone_geo
    zone.save()
    # None means that the centroid will be added in the geometric point of the zone
    # But we could provide a Shapely point as an alternative
    zone.add_centroid(None)


# %% md
## Centroid connectors

# %%
for zone_id, zone in zoning.all_zones().items():
    # We will connect for walk, with 1 connector per zone
    zone.connect_mode(mode_id="w", connectors=1)

    # And for cars, for cars with 2 connectors per zone
    # We also specify the link types we accept to connect to (can be used to avoid connection to ramps or freeways)
    zone.connect_mode(mode_id="c", link_types="ytrusP", connectors=2)

    # This takes a few minutes to compute, so we will break after processing the first 10 zones
    if zone_id >= 10:
        break

# %%
#  Let's add an special generator zones
# We also add a centroid at  the airport terminal
nodes = project.network.nodes

# Let's use some silly number for its ID, like 10,000, just so we can easily differentiate it
airport = nodes.new_centroid(10000)
airport.geometry = Point(166.91749582, -0.54472590)
airport.save()


# When connecting a centroid not associated with a zone, we need to tell AequilibraE what is the initial area around
# the centroid that needs to be considered when looking for candidate nodes
# Distance here is in degrees, so 0.01 is equivalent to roughly 1.1km
airport.connect_mode(airport.geometry.buffer(0.01), mode_id="c", link_types="ytrusP", connectors=1)


# %%


# %%
project.close()
