"""
Editing network geometry: Nodes
===============================

On this example we show how to mode a node in the network and look into
what happens to the links.
"""

## Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae.utils.create_example import create_example
from shapely.geometry import Point
import matplotlib.pyplot as plt

# We create the example project inside our temp folder
fldr = join(gettempdir(), uuid4().hex)

project = create_example(fldr)

# %%
# Let's see what the network looks like
# We don't recommend plotting it this way for real networks
links = project.network.links

# We plot the entire network
curr = project.conn.cursor()
curr.execute('Select link_id from links;')

for lid in curr.fetchall():
    geo = links.get(lid[0]).geometry
    plt.plot(*geo.xy, color='red')

# We also add the node we want to move
all_nodes = project.network.nodes
# We can just get one link in specific
node = all_nodes.get(1)
plt.plot(*node.geometry.xy, 'ro', color='green')

plt.show()
# %%
# Let's move node one from the upper left corner of the image above, a bit to the left and to the bottom




new_geo = Point(node.geometry.x + 0.02, node.geometry.y - 0.02)
node.geometry = new_geo

# We can save changes for all nodes we have edited so far
node.save()

# %%
# If you want to show the path in Python
# We do NOT recommend this, though....  It is very slow for real networks
# We plot the entire network
links.refresh()
curr = project.conn.cursor()
curr.execute('Select link_id from links;')

for lid in curr.fetchall():
    geo = links.get(lid[0]).geometry
    plt.plot(*geo.xy, color='blue')

plt.plot(*node.geometry.xy, 'ro', color='black')

plt.show()

# Did you notice the links shifted to match the node move?


# %%

project.close()
