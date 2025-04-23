"""
.. _example_transit_asssignment:

Public transport assignment with Optimal Strategies
===================================================

In this example, perform a Spiess & Florian assignment. `Click here <https://doi.org/10.1016/0191-2615(89)90034-9>`_
to check out the paper.

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
from uuid import uuid4
from os.path import join
from tempfile import gettempdir

from aequilibrae.transit import Transit
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

# %%
graph = data.create_graph(with_outer_stop_transfers=False, with_walking_edges=False, blocking_centroid_flows=False, connector_method="overlapping_regions")

# We drop geometry here for the sake of display.
graph.vertices.drop(columns="geometry")

# %%
graph.edges

# %%
# The graphs also also stored in the ``Transit.graphs`` dictionary. They are keyed by the 'period_id' they 
# were created for. A graph for a different 'period_id' can be created by providing ``period_id=`` in the 
# ``Transit.create_graph`` call. You can view previously created periods with the ``Periods`` object.
periods = project.network.periods
periods.data

# %%
# Connector project matching
# --------------------------
project.network.build_graphs()

# %%
# Now we'll create the line strings for the access connectors, this step is optinal but provides more accurate distance 
# estimations and better looking geometry.
# 
# Because Coquimbo doesn't have many walking edges we'll match onto the ``"c"`` graph.
graph.create_line_geometry(method="connector project match", graph="c")

# %%
# Saving and reloading
# --------------------
# Lets save all graphs to the 'public_transport.sqlite' database.
data.save_graphs()

# %%
# We can reload the saved graphs with ``data.load``. 
# This will create new ``TransitGraphBuilder``\'s based on the 'period_id' of the saved graphs.
# The graph configuration is stored in the 'transit_graph_config' table in 'project_database.sqlite' 
# as serialised JSON.
data.load()

# %%
# We can also remove the previously saved graphs.

# data.remove_graphs()

# %%
# Links and nodes are stored in a similar manner to the 'project_database.sqlite' database.
# 
# Reading back into AequilibraE
# -----------------------------
# You can create back in a particular graph via it's 'period_id'.
from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit.transit_graph_builder import TransitGraphBuilder

# %%
pt_con = database_connection("transit")

graph_db = TransitGraphBuilder.from_db(pt_con, periods.default_period.period_id)
graph_db.vertices.drop(columns="geometry")

# %%
graph_db.edges

# %%
# Converting to a AequilibraE graph object
# ----------------------------------------
# To perform an assignment we need to convert the graph builder into a graph.
transit_graph = graph.to_transit_graph()

# %%
# Mock demand matrix
# ------------------
# We'll create a mock demand matrix with demand 1 for every zone.
# We'll also need to convert from ``zone_id``\'s to ``node_id``\'s.
import numpy as np
from aequilibrae.matrix import AequilibraeMatrix

# %%
zones_in_the_model = len(transit_graph.centroids)

names_list = ['pt']

mat = AequilibraeMatrix()
mat.create_empty(zones=zones_in_the_model,
                 matrix_names=names_list,
                 memory_only=True)
mat.index = transit_graph.centroids[:]
mat.matrices[:, :, 0] = np.full((zones_in_the_model, zones_in_the_model), 1.0)
mat.computational_view()

# %%
# Hyperpath generation/assignment
# -------------------------------
# We'll create a ``TransitAssignment`` object as well as a ``TransitClass``
from aequilibrae.paths import TransitAssignment, TransitClass

# %%

# Create the assignment class
assigclass = TransitClass(name="pt", graph=transit_graph, matrix=mat)

assig = TransitAssignment()

assig.add_class(assigclass)

# We need to tell AequilbraE where to find the appropriate fields we want to use,  
# as well as the assignment algorithm to use.
assig.set_time_field("trav_time")
assig.set_frequency_field("freq")

assig.set_algorithm("os")

# When there's multiple matrix cores we'll also need to set the core to use for the demand.
assigclass.set_demand_matrix_core("pt")

# Let's perform the assignment for the transit classes added
assig.execute()

# %%
# View the results
assig.results()

# %%
# Saving results
# --------------
# We'll be saving the results to another sqlite db called 'results_database.sqlite'. 
# The 'results' table with 'project_database.sqlite' contains some metadata about each table in 
# 'results_database.sqlite'.
assig.save_results(table_name='hyperpath example')

# %%
# Wrapping up
project.close()
