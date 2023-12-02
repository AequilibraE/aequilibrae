"""
.. _example_gtfs_import_and_asssignment:

Public transport Spiess & Florian assignment
===========

In this example, we import a GTFS feed to our model, create a public transport network, create project match connectors, and perform a Spiess & Florian assignment.

We use data from Coquimbo, a city in La Serena Metropolitan Area in Chile.
"""
# %%
# Imports for example construction
from uuid import uuid4
from os import remove
from os.path import join
from tempfile import gettempdir
from aequilibrae.utils.create_example import create_example

# Imports for GTFS import
from aequilibrae.transit import Transit

# Imports for SF transit graph construction
import aequilibrae.transit.transit_graph_builder
from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit.transit_graph_builder import SF_graph_builder
from aequilibrae.project import Project

# Import for the Spiess & Florian assignment
from aequilibrae.paths.public_transport import HyperpathGenerating

# %%
# Let's create an empty project on an arbitrary folder.
fldr = join(gettempdir(), uuid4().hex)

project = create_example(fldr, "coquimbo")

# %%
# As the Coquimbo example already has a complete GTFS model, we shall remove its public transport
# database for the sake of this example.
remove(join(fldr, "public_transport.sqlite"))

# %%
# Let's import the GTFS feed.
dest_path = join(fldr, "gtfs_coquimbo.zip")


# %%
# Now we create our Transit object and import the GTFS feed into our model.
# This will automatically create a new public transport database.
data = Transit(project)

transit = data.new_gtfs_builder(agency="LISANCO", file_path=dest_path)

# %%
# To load the data, we must choose one date. We're going to continue with 2016-04-13 but feel free
# to experiment with any other available dates. Transit class has a function allowing you to check
# dates for the GTFS feed. It should take approximately 2 minutes to load the data.
transit.load_date("2016-04-13")

# %%
# Let's save this model for later use.
transit.save_to_disk()

# %%
# Graph building
# --------------
# Let's build the transit network. We'll disable `outer_stop_transfers` and `walking_edges` because Coquimbo doesn't have any parent stations.
pt_con = database_connection("transit")

graph = SF_graph_builder(pt_con, with_outer_stop_transfers=False, with_walking_edges=False, projected_crs="EPSG:4326")

# %%
# Creating the verticies and edges should only take a moment. By default zoning information is pulled from the project network. If you have your own zoning information add it using `graph.add_zones(zones)` before creating the verticies. We drop gemoetry here for the sake of display
graph.create_vertices()
graph.vertices.drop(columns="geometry")

# %%
graph.create_edges()
graph.edges

# %%
# Connector project matching
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
project.network.build_graphs()

# %%
# Now we'll create the line strings for the access connectors, this step is optinal but provides more accurate distance estimations and better looking geometry. Because Coquimbo doesn't have many walking edges we'll match onto the `"c"` graph.
graph.create_line_geometry(method="connector project match", graph="c")

# %%
# Saving
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Lets save this to the `public_transport.sqlite` database.
graph.save()

# %%
# Links and nodes are stored in a similar manner to the `project_database.sqlite` database.

# %%
# Reading back into AequilibraE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
graph_db = SF_graph_builder.from_db(pt_con)
graph_db.vertices.drop(columns="geometry")

# %%
graph_db.edges

# %%
# Converting to a AequilibraE graph object
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
g = graph.to_aeq_graph()

# %%
# Spiess & Florian assignment
# ---------------------------

# %%
# Mock demand matrix
# ~~~~~~~~~~~~~~~~~~
# We'll create a mock demand martix with demand `1` for every zone.
# We'll also need to convert from `zone_id`s to `node_id`s.
demand_df = graph.zones[["zone_id"]].merge(graph.zones.zone_id, how='cross')
demand_df["demand"] = 1.0
demand_df = graph.convert_demand_matrix_from_zone_to_node_ids(demand_df, o_zone_col="zone_id_x", d_zone_col="zone_id_y")
demand_df

# %%
# Hyperpath generation/assignment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We'll create a `HyperpathGenerating` obejct, providing the `SF` graph and field keys
hp = HyperpathGenerating(
    graph.edges, head="a_node", tail="b_node", trav_time="trav_time", freq="freq"
)

# %%
# Lets perform the assignment with the mock demand matrx
hp.assign(
    demand_df, origin_column="o_node_id", destination_column="d_node_id", demand_column="demand", check_demand=True
)

# %%
# View the results
hp._edges

# %%
# Saving results
# ~~~~~~~~~~~~~~
# We'll be saving the results to another sqlite db called `results_database.sqlite`. The `results` table with `project_database.sqlite` contains some metadata about each table in `results_database.sqlite`.
hp.save_results("hyperpath_results_demo", project=project)
hp.save_results("hyperpath_results_demo_without_zeros", keep_zero_flows=False, project=project)

# %%
# Wrapping up
project.close()
