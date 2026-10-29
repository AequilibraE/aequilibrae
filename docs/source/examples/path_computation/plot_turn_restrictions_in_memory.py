"""
.. _example_usage_path_computation_turn_restrictions:

Path computation with in-memory turn restrictions
=================================================

In this example, we show how to apply turn restrictions directly to a graph already loaded in memory.

We use the Coquimbo example and compare paths before and after prohibiting one specific turn.
"""

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.paths.graph.Graph.set_turn_restrictions`
#     * :func:`aequilibrae.paths.graph.Graph.compute_path`

# %%
# Imports
from os.path import join
from tempfile import gettempdir
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aequilibrae.utils.create_example import create_example

# %%
# We create the example project inside our temp folder
fldr = join(gettempdir(), uuid4().hex)
project = create_example(fldr, "coquimbo")

# %%
# We build all graphs and get the car graph.
project.network.build_graphs()
graph = project.network.graphs["c"]

# %%
# We set the cost field and disable centroid-flow blocking for this demonstration.
graph.set_graph("distance")
graph.set_blocked_centroid_flows(False)

# %%
# We choose origin/destination.
origin = 32343
destination = 22041

# %%
# Path without turn restrictions.
res_before = graph.compute_path(origin, destination)

# %%
# Build prohibited turns from node triples sampled along the baseline path.
path_nodes = [int(x) for x in res_before.path_nodes]
triples = [(path_nodes[i], path_nodes[i + 1], path_nodes[i + 2]) for i in range(max(0, len(path_nodes) - 2))]
turn_restrictions = pd.DataFrame(
    {
        "from_node": [t[0] for t in triples[:4]],
        "via_node": [t[1] for t in triples[:4]],
        "to_node": [t[2] for t in triples[:4]],
        "penalty": [np.nan, np.nan, np.nan, np.nan],
    }
)

# %%
# Apply turn restrictions in memory, with node-based U-turns globally prohibited
# (i.e. transitions that return to the previous node).
graph.set_turn_restrictions(turn_restrictions, allow_path_uturns=False)

# %%
# Path with the custom turn prohibition.
res_after = graph.compute_path(origin, destination)

# %%
# Plot both paths.
links = project.network.links.data.set_index("link_id")
links_before = links.loc[res_before.path]
links_after = links.loc[res_after.path]

# %%
m = links_before.explore(color="blue", style_kwds={'weight':5})
m = links_after.explore(m=m, color="red", style_kwds={'weight':5})

#%
m

# %%
# Close the project
project.close()
