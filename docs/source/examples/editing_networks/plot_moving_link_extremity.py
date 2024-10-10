"""
.. _editing_network_links:

Editing network geometry: Links
===============================

In this example, we move a link extremity from one point to another
and see what happens to the network.
"""
# %%
# .. admonition:: References
# 
#   * :ref:`modifications_on_links_layer` 

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.project.network.Links`

# %%

# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae.utils.create_example import create_example
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt

# %%

# We create the example project inside our temp folder
fldr = join(gettempdir(), uuid4().hex)

project = create_example(fldr)

# %%
all_nodes = project.network.nodes
links = project.network.links

# %%
# Let's move node one from the upper left corner of the image above, a bit to the left and to the bottom

# We edit the link that goes from node 1 to node 2
link = links.get(1)
node = all_nodes.get(1)
new_extremity = Point(node.geometry.x + 0.02, node.geometry.y - 0.02)
link.geometry = LineString([node.geometry, new_extremity])

# and the link that goes from node 2 to node 1
link = links.get(3)
node2 = all_nodes.get(2)
link.geometry = LineString([new_extremity, node2.geometry])

# We save the changes and refresh the links in memory for usage
links.save()
links.refresh()

# %%
# Because each link is unidirectional, you can no longer go from node 1 to node 2, obviously.
#
# We do NOT recommend this, though.... It is very slow for real networks.

# We plot the entire network.
curr = project.conn.cursor()
curr.execute("Select link_id from links;")

for lid in curr.fetchall():
    geo = links.get(lid[0]).geometry
    plt.plot(*geo.xy, color="blue")

all_nodes = project.network.nodes
curr = project.conn.cursor()
curr.execute("Select node_id from nodes;")

for nid in curr.fetchall():
    geo = all_nodes.get(nid[0]).geometry
    plt.plot(*geo.xy, "o", color="black")

plt.show()

# %%
# Now look at the network and how it used to be.

# %%
project.close()
