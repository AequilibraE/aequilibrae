"""
.. _plot_centroid_connection:

Create centroid connectors
==========================

We use Coquimbo example to show how we can create centroids and connect them to
the existing network efficiently.

We use Folium to visualize the resulting network.
"""

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.project.zoning`

# %%

# Imports
import folium
from uuid import uuid4
from tempfile import gettempdir
from os.path import join

from aequilibrae.utils.create_example import create_example

# sphinx_gallery_thumbnail_path = '../source/_images/plot_centroid_connector.png'

# %%

# Let's create an arbitrary project folder
fldr = join(gettempdir(), uuid4().hex)

# And create our Coquimbo project
project = create_example(fldr, "coquimbo")

# %%
# As Coquimbo already has centroids and centroid connectors, we should remove
# them for the sake of this example.
with project.db_connection as conn:
    conn.execute("DELETE FROM links WHERE name LIKE 'centroid connector%'")
    conn.execute("DELETE FROM nodes WHERE is_centroid=1;")
    conn.commit()

    centroids = conn.execute("SELECT COUNT(node_id) FROM nodes WHERE is_centroid=1;").fetchone()[0]

# %%
# If you want to make sure your deletion process has worked, you can check the
# number of centroids!
print("Current number of centroids: ", centroids)

# %%
zoning = project.zoning

# %%
# This centroid connector creation is effective because it uses the existing
# zone layer to create the centroids and connect them to the existing network.
#
# First thing to do is add the centroids to all zones that doesn't have a centroid
# at the geographic centroid of the zone, using ``add_centroids()``, which has a
# ``robust`` argument set to ``True`` as default. This means that it will automatically
# move the centroid location around to avoid conflicts with existing nodes.
zoning.add_centroids()

# %%
# Let's connect the mode ``c``, that stands for car.
mode = "c"

# %%
# Then we connect the centroid to the network, by selecting the desired mode,
# the number of connectors, the allowed link types for connection, and if one
# wants to allow the connection to links in other zones. By setting ``limit_to_zone=False``,
# we allow the centroid of one zone to be connected to a link outside the zone itself.

zoning.connect_mode(mode_id=mode, connectors=1, limit_to_zone=False)

# %%
# It is possible to repeat the process above for a different mode, with different
# link type, number of connectors and connection allowance, as desired.

# %%
# Finally, let's plot our data!

links = project.network.links.data
centroids = links[links["link_type"] == "centroid_connector"]
links = links[links["link_type"] != "centroid_connector"]

nodes = project.network.nodes.data
nodes = nodes[nodes["is_centroid"] == 1]

# %%
map = folium.Map(location=[-29.9568, -71.3456], zoom_start=14)
zoning.data.explore(m=map, color="blue", style_kwds={"fillOpacity": 0.05}, name="zones")
centroids.explore(m=map, color="black", style_kwds={"weight": 2.5}, name="centroid_connector")
links.explore(m=map, color="gray", style_kwds={"weight": 1}, name="links")
nodes.explore(m=map, color="red", style_kwds={"radius": 3, "fillOpacity": 1.0}, name="centroid")

folium.LayerControl().add_to(map)

map

# %%
project.close()

