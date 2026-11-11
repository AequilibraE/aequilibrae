"""
.. _example_transit_skimming:

Public transport assignment with skimming
=========================================

In this example, we build on the transit assignment example and add skimming to it.

We use data from Coquimbo, a city in La Serena Metropolitan Area in Chile.
"""
# %%
# .. admonition:: References
#
#   WE HIGHLY RECOMMEND YOU READ THE DOCUMENTATION ON SKIMMING BEFORE PROCEEDING
#
#   * :doc:`../../public_transport`

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.transit.transit`
#     * :func:`aequilibrae.transit.transit_graph_builder`
#     * :func:`aequilibrae.paths.traffic_class.TransitClass`
#     * :func:`aequilibrae.paths.traffic_assignment.TransitAssignment`
#     * :func:`aequilibrae.matrix.aequilibrae_matrix`

# %%
# Imports for example construction
from os.path import join
from tempfile import gettempdir
from uuid import uuid4

import numpy as np

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import TransitAssignment, TransitClass, TransitCosting
from aequilibrae.transit import Transit
from aequilibrae.transit.transit_graph_builder import TransitGraphBuilder
from aequilibrae.utils.create_example import create_example

# sphinx_gallery_thumbnail_path = '../source/_images/transit/skimming.png'

# %%

# Let's create an empty project on an arbitrary folder.
fldr = join(gettempdir(), uuid4().hex)

project = create_example(fldr, "coquimbo")

# %%
# Let's create our ``Transit`` object.
data = Transit(project)

# %%
# Graph building
# --------------
# Let's build the transit network. We'll disable ``outer_stop_transfers`` and ``walking_edges``
# because Coquimbo doesn't have any parent stations.
#
# For the OD connections we'll use the ``overlapping_regions`` method and create some accurate line geometry later.
# Creating the graph should only take a moment. By default zoning information is pulled from the project network.
# If you have your own zoning information add it using ``graph.add_zones(zones)`` then ``graph.create_graph()``.

graph = data.create_graph(
    with_outer_stop_transfers=False,
    with_walking_edges=False,
    blocking_centroid_flows=False,
    connector_method="overlapping_regions"
)

# %%
# Connector project matching
# --------------------------
graph.create_line_geometry(method="connector project match", graph="c")
data.save_graphs()
data.load()

# Reading back into AequilibraE
graph_db = TransitGraphBuilder.from_db(project, project.network.periods.default_period.period_id)
graph_db.vertices.drop(columns="geometry")

# To perform an assignment we need to convert the graph builder into a graph.
transit_graph = graph_db.to_transit_graph()

# %%

# Mock demand matrix
zones = len(transit_graph.centroids)
mat = AequilibraeMatrix(index=transit_graph.centroids, index_name="zones")
mat["pt"] = np.full((zones, zones), 1.0)

# %%
# Hyperpath generation/assignment
# -------------------------------
# We'll create a ``TransitAssignment`` object as well as a ``TransitClass``, whose costing waits half a headway.

# %%

# Create the assignment class
assigclass = TransitClass(name="pt", graph=transit_graph, matrix=mat, costing=TransitCosting(headway_fraction=0.5))

assig = TransitAssignment()

assig.add_class(assigclass)

# Set assignment
assig.set_time_field("trav_time")
assig.set_frequency_field("freq")
assig.set_skimming_fields(
    ["trav_time", "boardings", "transfers", "in_vehicle_trav_time", "waiting_time", "first_waiting_time"]
)
assig.set_algorithm("os")

# Perform the assignment for the transit classes added
assig.execute()

# %%
# We can use the get_skim_results() method to retrieve the skims. Waits are the headway fraction over the combined
# frequency of the lines boarded at each stop, and transfers are boardings beyond the first.
skims = assig.get_skim_results()["pt"]

# %%
# Pairs of zones that no strategy connects skim as infinite, rather than zero. In Coquimbo, every zone connects to
# a stop, so transit reaches every pair.
reachable = np.isfinite(skims["trav_time"])
reachable.mean()

# %%
# Mean waits and transfers over the pairs that transit connects
skims["waiting_time"][reachable].mean(), skims["transfers"][reachable].mean()

# %%
# Saving results
# --------------
# We'll be saving the skimming results to the project's matrices, which records the period, graph and costing
# they came from.
assig.save_skims("hyperpath_example")

# %%
# The link volumes go to the results database.
assig.save_results(table_name="hyperpath example")

# %%
# Wrapping up
project.close()
