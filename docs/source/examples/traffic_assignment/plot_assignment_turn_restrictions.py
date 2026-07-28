"""
.. _plot_assignment_turn_restrictions:

Traffic assignment with turn penalty and prohibition
====================================================

In this example, we run static traffic assignment for Sioux Falls in two scenarios:

1. Base case (no custom turn restrictions)
2. Custom turn controls with:
   - a turn penalty from link 7 to link 36
   - a turn prohibition from link 24 to link 23

All turn directions are 1 and node-based U-turn transitions are allowed.
"""

# %%
# .. admonition:: References
#
#   * :doc:`../../static_traffic_assignment`

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.paths.graph.Graph.set_turn_restrictions`
#     * :func:`aequilibrae.paths.traffic_class.TrafficClass`
#     * :func:`aequilibrae.paths.traffic_assignment.TrafficAssignment`

# %%
# Imports
from os.path import join
from tempfile import gettempdir
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aequilibrae.paths import TrafficAssignment
from aequilibrae.paths.traffic_class import TrafficClass
from aequilibrae.utils.create_example import create_example


def run_assignment(graph, matrix):
    """Runs a single-class traffic assignment and returns link flow results."""
    traffic_class = TrafficClass("car", graph, matrix)

    assignment = TrafficAssignment()
    assignment.set_classes([traffic_class])
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("free_flow_time")
    assignment.set_algorithm("bfw")
    assignment.max_iter = 100
    assignment.rgap_target = 1e-2
    assignment.execute()

    return assignment.results()


# %%
# We create the Sioux Falls example project inside our temp folder.
fldr = join(gettempdir(), uuid4().hex)
project = create_example(fldr, "sioux_falls")

# %%
# Build graphs and load demand matrix.
project.network.build_graphs()
graph = project.network.graphs["c"]

graph.set_graph("free_flow_time")
graph.set_blocked_centroid_flows(False)

matrix = project.matrices.get_matrix("demand_omx")
matrix.computational_view(["matrix"])

# %%
# Base assignment.
base_flows = run_assignment(graph, matrix)

# %%
# Apply turn controls in memory as node triples equivalent to:
# - a turn penalty from link 7 to link 36 (dir=1)
# - a turn prohibition from link 24 to link 23 (dir=1)
links = project.network.links.data.set_index("link_id")
turn_restrictions = pd.DataFrame(
    {
        "from_node": [int(links.loc[7, "a_node"]), int(links.loc[24, "a_node"])],
        "via_node": [int(links.loc[7, "b_node"]), int(links.loc[24, "b_node"])],
        "to_node": [int(links.loc[36, "b_node"]), int(links.loc[23, "b_node"])],
        "penalty": [5.0, np.nan],  # 5.0 minutes penalty, and one prohibited turn
    }
)
graph.set_turn_restrictions(turn_restrictions, allow_path_uturns=True)

# %%
# Assignment with custom turn controls.
restricted_flows = run_assignment(graph, matrix)

# %%
# Plot flows for both cases and the difference.
links = project.network.links.data.set_index("link_id")

plot_df = links[["geometry"]].join(
    base_flows[["matrix_tot"]].rename(columns={"matrix_tot": "base_flow"})
).join(
    restricted_flows[["matrix_tot"]].rename(columns={"matrix_tot": "restricted_flow"})
)
plot_df["difference"] = plot_df["restricted_flow"] - plot_df["base_flow"]

max_base = max(plot_df["base_flow"].max(), 1e-9)
max_restricted = max(plot_df["restricted_flow"].max(), 1e-9)
max_diff = max(np.abs(plot_df["difference"]).max(), 1e-9)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

plot_df.plot(
    ax=axes[0],
    linewidth=4 * plot_df["base_flow"] / max_base,
    color="tab:blue",
)
axes[0].set_title("Base assignment flow")
axes[0].set_axis_off()

plot_df.plot(
    ax=axes[1],
    linewidth=4 * plot_df["restricted_flow"] / max_restricted,
    color="tab:green",
)
axes[1].set_title("With turn penalty/prohibition")
axes[1].set_axis_off()

plot_df.plot(
    ax=axes[2],
    linewidth=3 * np.abs(plot_df["difference"]) / max_diff,
    color="lightgray",
)

plot_df.loc[plot_df["difference"] >= 0].plot(
    ax=axes[2],
    linewidth=3 * np.abs(plot_df.loc[plot_df["difference"] >= 0, "difference"]) / max_diff,
    color="tab:red",
)

plot_df.loc[plot_df["difference"] < 0].plot(
    ax=axes[2],
    linewidth=3 * np.abs(plot_df.loc[plot_df["difference"] < 0, "difference"]) / max_diff,
    color="tab:purple",
)

axes[2].set_title("Flow difference (restricted - base)")
axes[2].set_axis_off()

plt.tight_layout()

# %%
# Close matrix and project.
matrix.close()
project.close()
