"""
Toll analysis with Select Link
==============================

On this example we show how to perform traffic assignment considering fixed
link costs (e.g. tolls) besides travel time. We also do assignment with
three different vehicle classes
"""

import logging
import sys
import urllib
import zipfile
from os.path import join

## Imports
from tempfile import gettempdir

# %%
from aequilibrae import logger, Project
from aequilibrae.paths import TrafficAssignment, TrafficClass

# We the project open, we can tell the logger to direct all messages to the terminal as well
stdout_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s;%(name)s;%(levelname)s ; %(message)s")
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

# %%

disk_pth = join(gettempdir(), "sioux_falls_multi-Class")
urllib.request.urlretrieve("https://www.aequilibrae.com/data/sioux_falls_multi-Class.zip", disk_pth + ".zip")

with zipfile.ZipFile(disk_pth + ".zip") as zf:
    zf.extractall(gettempdir())

# %%
project = Project()
project.open(disk_pth)

# %%
proj_matrices = project.matrices
proj_matrices.update_database()
mat_list = proj_matrices.list()

# %%
# we build all graphs
project.network.build_graphs()
carGraph = project.network.graphs["c"]
truckGraph = project.network.graphs["T"]
motoGraph = project.network.graphs["M"]

# %%%

matrix_name = "demand_mc_omx"
carDemand = proj_matrices.get_matrix(matrix_name)
carDemand.computational_view("car")
carClass = TrafficClass("car", carGraph, carDemand)
carClass.set_pce(1)
carClass.set_vot(35)
carClass.set_fixed_cost("toll", 0.025)

# %%


motoDemand = proj_matrices.get_matrix(matrix_name)
motoDemand.computational_view("motorcycle")
motoClass = TrafficClass("motorcycle", motoGraph, motoDemand)
motoClass.set_pce(0.2)
motoClass.set_vot(35)
# Fixed cost can be any field (or field_AB/BA in the network).  And the factor defaults to 1.0
motoClass.set_fixed_cost("toll", 0.0125)

# %%
truckDemand = proj_matrices.get_matrix(matrix_name)
truckDemand.computational_view("trucks")
truckClass = TrafficClass("trucks", truckGraph, truckDemand)
truckClass.set_pce(1.5)
truckClass.set_vot(35)
truckClass.set_fixed_cost("toll", 0.05)

# %%

# OPTIONAL: If we want to execute select link analysis on a particular TrafficClass, we set the links we are analysing
# The format of the input select links is a dictionary (str: list[tuple]).
# Each entry represents a separate set of selected links to compute. The str name will name the set of links
# The list[tuple] is the list of links being selected, of the form (link_id, direction), as it occurs in the Graph
# direction can be 0, 1, -1. 0 denotes bi-directionality
# For example, let's use Select Link on four sets of links to see what OD pairs are using tolls

select_links = {
    "toll_link_37": [(37, 1)],
    "toll_link_24": [(24, 1)],
    "toll_link_56": [(56, 1)],
    "all_other_tolls": [(49, 1), (36, 1)],
}
# We call this command on the class we are analysing with our dictionary of values
# We can query one or more classes
carClass.set_select_links(select_links)
truckClass.set_select_links(select_links)


# %%
assig = TrafficAssignment()
# Different than some commercial software, your results are as proportional as the
# convergence models allow and your results do not depend on the order you put the classes in.
assig.set_classes([carClass, motoClass, truckClass])

assig.set_vdf("BPR")  # This is not case-sensitive # Then we set the volume delay function

assig.set_vdf_parameters({"alpha": "alpha", "beta": "beta"})  # And its parameters

assig.set_time_field("free_flow_time")
assig.set_capacity_field(f"capacity")  # The capacity and free flow travel times as they exist in the graph

# And the algorithm we want to use to assign
assig.set_algorithm("bfw")

# You would obviously pick a much tighter convergence criterium and correspondingly larger number of iterations
assig.max_iter = 30
assig.rgap_target = 0.01

assig.execute()  # we then execute the assignment
assig.save_results("test_assignment")

# %%
# Now let us save our select link results, all we need to do is provide it with a name
# In additional to exporting the select link flows, it also exports the Select Link matrices in OMX format.
assig.save_select_link_results("select_link_analysis")

# %%
# Say we just want to save our select link flows, we can call:
assig.save_select_link_flows("just_select_link_flows")

# Or if we just want the SL matrices:
assig.save_select_link_matrices("just_select_link_matrices")
# Internally, the save_select_link_results calls both of these methods at once.

# %% md
# Let's validate our results against those previously converged to 1 e-5
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd

charted = ["car", "motorcycle", "trucks", "pce"]
mod_res_path = join(disk_pth, "results_database.sqlite")

conn = sqlite3.connect(mod_res_path)
assig_results = pd.read_sql(f"Select * from test_assignment", conn)
ref_results = pd.read_sql(f"Select * from fully_converged", conn)
conn.close()

assig_results.set_index(["link_id"], inplace=True)
assig_results.columns = [x.lower() for x in assig_results.columns]

ref_results.set_index(["link_id"], inplace=True)
ref_results.columns = [x.lower() for x in ref_results.columns]

df = assig_results.join(ref_results, rsuffix="_ref")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

for per, ax in zip(charted, [ax1, ax2, ax3, ax4]):
    ax.scatter(df[f"{per}_tot"], df[f"{per}_tot_ref"])
    ax.set(title=per.upper())
    ax.grid()
    ax.set_xlabel("my flows")
    ax.set_ylabel("reference")

plt.show()
