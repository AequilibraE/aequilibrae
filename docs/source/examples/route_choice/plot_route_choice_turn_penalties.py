"""
.. _example_usage_route_choice_turn_penalties:

Route Choice with Turn Penalties
================================

In this example, we show how route choice accounts for turn penalties during
choice set generation and path-size logit calculations.
"""
# %%
# .. admonition:: References
#
#   * :doc:`../../route_choice`
#   * :ref:`turn_restrictions`

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.paths.graph.Graph.set_turn_restrictions`
#     * :func:`aequilibrae.paths.route_choice.RouteChoice`

# %%
# Imports
import numpy as np
import pandas as pd

from aequilibrae.paths import Graph, RouteChoice

# %%
# Create a small network with two routes from node 10 to node 40.
graph = Graph()
graph.network = pd.DataFrame(
    {
        "link_id": [71, 12, 55, 24],
        "a_node": [10, 20, 10, 30],
        "b_node": [20, 40, 30, 40],
        "direction": [1, 1, 1, 1],
        "time": [1.0, 1.0, 2.0, 2.0],
    }
)
graph.prepare_graph(np.array([10, 20, 30, 40]), remove_dead_ends=False)
graph.set_blocked_centroid_flows(False)
graph.set_graph("time")

# %%
# Add a five-unit turn penalty to the direct route.
graph.set_turn_restrictions(
    pd.DataFrame(
        {
            "from_node": [10],
            "via_node": [20],
            "to_node": [40],
            "penalty": [5.0],
        }
    )
)

# %%
# Generate the route choice set. The direct route has a cost of 2 plus the
# turn penalty, so the alternative route with a cost of 4 is more likely.
rc = RouteChoice(graph)
rc.set_choice_set_generation("bfsle", max_routes=2, max_depth=5)
rc.execute_single(10, 40, demand=1.0)

route_choice_results = rc.get_results()
route_choice_results

# %%
# We can also recalculate path-size logit results for routes imported from
# another source. Route link IDs use a positive sign for the AB direction and
# a negative sign for the BA direction.
imported_routes = pd.DataFrame(
    {
        "origin id": [10, 10],
        "destination id": [40, 40],
        "route set": [[71, 12], [55, 24]],
    }
)

# %%
# ``recompute_psl`` validates origin and destination endpoints, link
# connectivity and turn restrictions before calculating costs, overlap and
# probabilities.
imported_results = rc.recompute_psl(imported_routes)
imported_results
