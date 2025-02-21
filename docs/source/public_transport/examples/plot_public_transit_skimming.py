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
#   * :doc:`../../public_transport`

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.transit.Transit`
#     * :func:`aequilibrae.transit.TransitGraphBuilder`
#     * :func:`aequilibrae.paths.TransitClass`
#     * :func:`aequilibrae.paths.TransitAssignment`
#     * :func:`aequilibrae.matrix.AequilibraeMatrix`

# %%
# Imports for example construction
from os.path import join
from tempfile import gettempdir
from uuid import uuid4

import numpy as np

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import TransitAssignment, TransitClass
from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit import Transit
from aequilibrae.transit.transit_graph_builder import TransitGraphBuilder
from aequilibrae.utils.create_example import create_example

# sphinx_gallery_thumbnail_path = '../source/_images/transit/hyperpath_bell_n_10_alpha_100d0.png'

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
project.network.build_graphs()
graph.create_line_geometry(method="connector project match", graph="c")
data.save_graphs()
data.load()

# Reading back into AequilibraE
pt_con = database_connection("transit")
graph_db = TransitGraphBuilder.from_db(pt_con, project.network.periods.default_period.period_id)
graph_db.vertices.drop(columns="geometry")

# To perform an assignment we need to convert the graph builder into a graph.
transit_graph = graph.to_transit_graph()

# %%

# Mock demand matrix
zones = len(transit_graph.centroids)
mat = AequilibraeMatrix()
mat.create_empty(zones=zones, matrix_names=['pt'], memory_only=True)
mat.index = transit_graph.centroids[:]
mat.matrices[:, :, 0] = np.full((zones, zones), 1.0)
mat.computational_view()

# %%
# Hyperpath generation/assignment
# -------------------------------
# We'll create a ``TransitAssignment`` object as well as a ``TransitClass``.

# %%

# Create the assignment class
assigclass = TransitClass(name="pt", graph=transit_graph, matrix=mat)

assig = TransitAssignment()

assig.add_class(assigclass)

# Set assignment
assig.set_time_field("trav_time")
assig.set_frequency_field("freq")
assig.set_skimming_fields(["boardings"])
assig.set_algorithm("os")
assigclass.set_demand_matrix_core("pt")

# Perform the assignment for the transit classes added
assig.execute()

# We can use the get_skim_results() method to retrieve the skims
assig.get_skim_results()["pt"].matrix["boardings"].sum()

# %%
# Saving results
# --------------
# We'll be saving the skimming results.
assig.save_results(table_name='hyperpath example')

# %%
# Wrapping up
project.close()
