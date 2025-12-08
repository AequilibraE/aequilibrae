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
.. _find_disconnected_links:

Finding disconnected links
==========================

In this example, we show how to find disconnected links in an AequilibraE network.

We use the Nauru example to find disconnected links.
"""
# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.paths.results.path_results`

# %%

# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from datetime import datetime
import pandas as pd
import numpy as np
from aequilibrae.utils.create_example import create_example
from aequilibrae.paths.results import PathResults
# sphinx_gallery_thumbnail_path = '../source/_images/disconnected_network.png'

# %%

# We create an empty project on an arbitrary folder
fldr = join(gettempdir(), uuid4().hex)

# Let's use the Nauru example project for display
project = create_example(fldr, "nauru")

# Let's analyze the mode car or 'c' in our model
mode = "c"

# %%
# We need to create the graph, but before that, we need to have at least one centroid in our network.

# We get an arbitrary node to set as centroid and allow for the construction of graphs
nodes = project.network.nodes
centroid_count = nodes.data.query('is_centroid == 1').shape[0]

if centroid_count == 0:
    arbitrary_node = nodes.data["node_id"][0]
    nd = nodes.get(arbitrary_node)
    nd.is_centroid = 1
    nd.save()

network = project.network
network.build_graphs(modes=[mode])
graph = network.graphs[mode]
graph.set_blocked_centroid_flows(False)

if centroid_count == 0:
    # Let's revert to setting up that node as centroid in case we had to do it

    nd.is_centroid = 0
    nd.save()

# %%
# We set the graph for computation
graph.set_graph("distance")
graph.set_skimming("distance")

# %%
# Get the nodes that are part of the car network
missing_nodes = nodes.data.query("modes.str.contains(@mode)")["node_id"].values

# %%
# And prepare the path computation structure
res = PathResults()
res.prepare(graph)

# %%
# Now we can compute all the path islands we have

islands = []
idx_islands = 0

while missing_nodes.shape[0] >= 2:
    print(datetime.now().strftime("%H:%M:%S"), f" - Computing island: {idx_islands}")
    res.reset()
    res.compute_path(missing_nodes[0], missing_nodes[1])
    res.predecessors[graph.nodes_to_indices[missing_nodes[0]]] = 0
    connected = graph.all_nodes[np.where(res.predecessors >= 0)]
    connected = np.intersect1d(missing_nodes, connected)
    missing_nodes = np.setdiff1d(missing_nodes, connected)
    print(f"    Nodes to find: {missing_nodes.shape[0]:,}")
    df = pd.DataFrame({"node_id": connected, "island": idx_islands})
    islands.append(df)
    idx_islands += 1

print(f"\nWe found {idx_islands} islands")

# %%
# Let's consolidate everything into a single DataFrame
islands = pd.concat(islands)

# And save to disk alongside our model
islands.to_csv(join(fldr, "island_outputs_complete.csv"), index=False)

# %%
# If you join the ``node_id`` field in the CSV file generated above with the ``a_node`` or ``b_node`` 
# fields in the links table, you will have the corresponding links in each disjoint island found.

# %%
project.close()
