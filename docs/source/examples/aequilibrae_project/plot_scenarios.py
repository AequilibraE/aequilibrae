"""
.. _example_usage_scenarios:

Project Scenarios
=================

In this example, we show how to use AequilibraE's scenario system to manage multiple model variants
within a single project, using different example networks to demonstrate scenario isolation and management.
"""
# %%
# .. admonition:: References
#
#   * :doc:`../../aequilibrae_project`

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.project.project.Project.list_scenarios`
#     * :func:`aequilibrae.project.project.Project.use_scenario`
#     * :func:`aequilibrae.project.project.Project.create_empty_scenario`
#     * :func:`aequilibrae.project.project.Project.clone_scenario`

# %%

# Imports
from uuid import uuid4
from tempfile import gettempdir
from pathlib import Path

from aequilibrae.utils.create_example import create_example
from aequilibrae import TrafficAssignment, TrafficClass

# sphinx_gallery_thumbnail_path = '../source/_images/plot_scenarios.png'

# %%

# We create the example project inside our temp folder.
fldr = Path(gettempdir()) / uuid4().hex
project = create_example(fldr, "sioux_falls")


# %%
# Working with scenarios
# ----------------------
# Let's first see what scenarios exist in our project

project.list_scenarios()

# %%
# The root scenario is always present and represents the base model.
# Let's examine the current scenario's network

print(f"Current scenario network has {len(project.network.links.data)} links")
print(f"Current scenario network has {len(project.network.nodes.data)} nodes")

# %%
# Creating new scenarios
# ----------------------
# We can create empty scenarios or clone existing ones

# Create an empty scenario to manually populate with a future/different network
project.create_empty_scenario("test_modifications", "Scenario for testing network modifications")

# Clone the root scenario to preserve the original network
project.clone_scenario("limited_capacity", "Testing different assignment parameters")

# %%
# Let's see our updated scenario list

project.list_scenarios()

# %%
# Switching between scenarios
# ---------------------------
# Each scenario operates independently with its own data

# Switch to the cloned scenario
project.use_scenario("limited_capacity")
print(f"This scenario has {len(project.network.links.data)} links")

# Modify the network
with project.db_connection as conn:
    conn.execute("UPDATE links SET capacity_ab=capacity_ab/2, capacity_ba=capacity_ba/2 WHERE link_id > 20 AND link_id < 50")

# %%
# Let's perform a traffic assignment in this scenario with lowered capacity

# Build the network graph
project.network.build_graphs(fields=["distance", "capacity_ab", "capacity_ba"], modes=["c"])
graph = project.network.graphs["c"]
graph.set_graph("distance")
graph.set_blocked_centroid_flows(False)

# Get the demand matrix
mat = project.matrices.get_matrix("demand_omx")
mat.computational_view()

# Create traffic assignment with alternative parameters
assigclass = TrafficClass("car", graph, mat)
assignment = TrafficAssignment(project)
assignment.add_class(assigclass)
assignment.set_vdf("BPR")

assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
assignment.set_capacity_field("capacity")
assignment.set_time_field("distance")
assignment.max_iter = 10
assignment.set_algorithm("msa")

assignment.execute()

# Save results specific to this scenario
assignment.save_results("alternative_assignment")

print(f"Assignment completed. Total flow: {assigclass.results.total_link_loads.sum():.2f}")

# %%
# Switch to empty scenario for modifications
project.use_scenario("test_modifications")
print(f"Empty scenario has {len(project.network.links.data)} links")

# This scenario starts with an empty network, suitable for building from scratch
# or testing specific network configurations

# %%
# Scenario isolation demonstration
# --------------------------------
# Let's switch back to root and show that scenarios are isolated

project.use_scenario("root")
print(f"Back to root scenario with {len(project.network.links.data)} links")

# Check results - only root scenario results should be visible
root_results = project.results.list()
print(f"Root scenario has {len(root_results)} result tables")

# Switch to alternative scenario and check its results
project.use_scenario("limited_capacity")
alt_results = project.results.list()
print(f"Alternative scenario has {len(alt_results)} result tables")

# Each scenario maintains its own results database
alternative_assignment_exists = "alternative_assignment" in alt_results["table_name"].values
print(f"Alternative assignment result exists in this scenario: {alternative_assignment_exists}")

# %%
# Best practices for scenario management
# --------------------------------------

# Always return to root when doing project-wide operations
project.use_scenario("root")

# List scenarios for reference
final_scenarios = project.list_scenarios()
print("\nFinal scenario summary:")
for _, scenario in final_scenarios.iterrows():
    project.use_scenario(scenario['scenario_name'])
    link_count = len(project.network.links.data)
    result_count = len(project.results.list())
    print(f"  {scenario['scenario_name']}: {link_count} links, {result_count} results")
    print(f"    Description: {scenario['description']}")

# %%
# Clean up
project.use_scenario("root")  # Always end on root scenario
mat.close()
project.close()
