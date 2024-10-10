"""
.. _editing_network_nodes:

Editing network geometry: Nodes
===============================

In this example, we show how to mode a node in the network and look into
what happens to the links.
"""
# %%
# .. admonition:: References
# 
#   * :ref:`modifications_on_nodes_layer` 

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.project.network.Nodes`

# %%

# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae.utils.create_example import create_example
from shapely.geometry import Point
import matplotlib.pyplot as plt

# %%

# We create the example project inside our temp folder.
fldr = join(gettempdir(), uuid4().hex)

project = create_example(fldr)

# %%
# Let's move node one from the upper left corner of the image above, a bit to the left and to the bottom.

# We also add the node we want to move.
all_nodes = project.network.nodes
links = project.network.links
node = all_nodes.get(1)
new_geo = Point(node.geometry.x + 0.02, node.geometry.y - 0.02)
node.geometry = new_geo

# We can save changes for all nodes we have edited so far.
node.save()

# %%
# If you want to show the path in Python.
# 
# We do NOT recommend this, though.... It is very slow for real networks.

# Let's refresh the links in memory for usage
links.refresh()

curr = project.conn.cursor()
curr.execute("Select link_id from links;")

# We plot the entire network.
for lid in curr.fetchall():
    geo = links.get(lid[0]).geometry
    plt.plot(*geo.xy, color="blue")

plt.plot(*node.geometry.xy, "o", color="black")

plt.show()

# %%
# Did you notice the links are matching the node?
# Look at the original network and see how it used to look like.

# %%
project.close()
