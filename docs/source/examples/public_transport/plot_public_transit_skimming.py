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
#     * :class:`aequilibrae.transit.Transit`
#     * :class:`aequilibrae.builder.TransitGraphBuilder`
#     * :class:`aequilibrae.transit.TransitClass`
#     * :class:`aequilibrae.transit.TransitAssignment`
#     * :class:`aequilibrae.matrix.AequilibraeMatrix`

# %%
# Imports for example construction
from os.path import join
from tempfile import gettempdir
from uuid import uuid4

import numpy as np

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.transit import TransitAssignment, TransitClass
from aequilibrae.transit import Transit
from aequilibrae.builder import TransitGraphBuilder
from aequilibrae.examples import create_example

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
graph_db = TransitGraphBuilder.from_db(project, project.tables.periods.default_period.period_id)
graph_db.vertices.drop(columns="geometry")

# To perform an assignment we need to convert the graph builder into a graph.
transit_graph = graph_db.to_transit_graph()
transit_graph.set_skimming_fields(["trav_time", "boardings", "freq"])

# %%

# Mock demand matrix
zones = len(transit_graph.centroids)
mat = AequilibraeMatrix(index=transit_graph.centroids, index_name="zones")
mat["pt"] = np.full((zones, zones), 1.0)

# %%
# Hyperpath generation/assignment
# -------------------------------
# We'll create a ``TransitAssignment`` object as well as a ``TransitClass``.

# %%

# Create the assignment class
assigclass = TransitClass(name="pt", graph=transit_graph, matrix=mat)

assig = TransitAssignment(project=project)

assig.add_class(assigclass)

# Set assignment
assig.set_time_field("trav_time")
assig.set_frequency_field("freq")
assig.set_algorithm("os")

# Perform the assignment for the transit classes added
assig.execute()

# We can use the get_skim_results() method to retrieve the skims
assig.get_skim_results()["pt"]["boardings"].sum()

# %%
# Saving results
# --------------
# We'll save the assignment's link-volume results to results_database.sqlite.
# The skim matrices returned above remain in memory; this call does not save them.
assig.save_results(table_name='hyperpath example')

# %%
# Wrapping up
project.close()
