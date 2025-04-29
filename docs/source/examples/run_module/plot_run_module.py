"""
.. _run_module_example:

Run module
==========

In this example we demonstrate how to use AequilibraE's run module using Sioux Falls example.
"""
# %%
# .. admonition:: References
# 
#   * :doc:`../../run_module`
#   * :ref:`parameters_run`

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :attr:`aequilibrae.project.Project.run`

# %%

# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join

from aequilibrae.parameters import Parameters
from aequilibrae.utils.create_example import create_example

# %%

# Let's create the Sioux Falls example in an arbitrary folder.
folder = join(gettempdir(), uuid4().hex)

project = create_example(folder)

# %%
# First, let's check the matrix information using ``matrix_summary()``. This method
# provides us useful information such as the matrix total, minimum and maximum values
# in the array, and the number of non-empty pairs in the matrix.
#
# Notice that the matrix summary is presented for each matrix core.
project.run.matrix_summary()

# %%
# If our matrices folder is empty, instead of a nested dictionary of data,
# AequilibraE run would return an empty dictionary.

# %%
# Let's create a graph for mode `car`.
mode = "c"

# %%
network = project.network
network.build_graphs(modes=[mode])
graph = network.graphs[mode]
graph.set_graph("distance")
graph.set_skimming("distance")
graph.set_blocked_centroid_flows(False)

# %%
# With the method ``graph_summary()``, we can check the total number of links, nodes, and zones,
# as well as the compact number of links and nodes used for computation. If we had more than
# one graph, its information would be displayed within the nested dictionary.

project.run.graph_summary()

# %%
# If no graphs have been built, an empty dictionary will be returned.

# %%
# Let's add a ``create_delaunay`` function to our ``run/__init__.py`` file.
#
# This function replicates the example in which we :ref:`create Delaunay lines <creating_delaunay_lines>`.
func_string = """
def create_delaunay(source: str, name: str, computational_view: str, result_name: str, overwrite: bool=False):\n
\tfrom aequilibrae.utils.create_delaunay_network import DelaunayAnalysis\n
\tproject = get_active_project()\n
\tmatrix = project.matrices\n
\tmat = matrix.get_matrix(name)\n
\tmat.computational_view(computational_view)\n
\tda = DelaunayAnalysis(project)\n
\tda.create_network(source, overwrite)\n
\tda.assign_matrix(mat, result_name)\n
"""

# %%
with open(join(folder, "run", "__init__.py"), "a") as file:
    file.write("\n")
    file.write(func_string)

# %%
# Now we add new parameters to our model

p = Parameters(project)
p.parameters["run"]["create_delaunay"] = {}
p.parameters["run"]["create_delaunay"]["source"] = "zones"
p.parameters["run"]["create_delaunay"]["name"] = "demand_omx"
p.parameters["run"]["create_delaunay"]["computational_view"] = "matrix"
p.parameters["run"]["create_delaunay"]["result_name"] = "my_run_module_example"
p.write_back()

# %%
# And we run the function
project.run.create_delaunay()

# %%
# .. note::
#
#    To run the ``create_delaunay`` function we created above without argument
#    values, we must insert the values as a project parameter. Adding an unused
#    parameter to the ``parameters.yml`` file will raise an execution error.
#

# %%
# Creating Delaunay lines also creates a ``results_database.sqlite`` that contains the
# result of the all-or-nothing algorithim that generated the output. We can check
# the existing results in the results_database using the ``results_summary`` method.
project.run.results_summary()

# %%
project.close()
