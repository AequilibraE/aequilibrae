"""
Toll analysis
=============

On this example we show how to perform traffic assignment considering fixed
link costs (e.g. tolls) besides data
"""

## Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
import urllib
import zipfile

# %%
from aequilibrae import logger, Project
from aequilibrae.paths import TrafficAssignment, TrafficClass
import logging
import sys

# We the project open, we can tell the logger to direct all messages to the terminal as well
stdout_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s;%(name)s;%(levelname)s ; %(message)s")
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

# %%

disk_pth = join(gettempdir(), 'sioux_falls_multi-Class')
urllib.request.urlretrieve('https://www.aequilibrae.com/data/sioux_falls_multi-Class.zip', disk_pth + '.zip')

with zipfile.ZipFile(disk_pth + '.zip') as zf:
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
carGraph = project.network.graphs['c']
truckGraph = project.network.graphs['T']
motoGraph = project.network.graphs['M']

# %%%

matrix_name = 'demand_mc_omx'
carDemand = proj_matrices.get_matrix(matrix_name)
carDemand.computational_view('car')
carClass = TrafficClass(carGraph, carDemand)
carClass.set_pce(1)
carClass.set_vot(35)
carClass.set_fixed_cost('toll', 0.025)

# %%


motoDemand = proj_matrices.get_matrix(matrix_name)
motoDemand.computational_view('motorcycle')
motoClass = TrafficClass(motoGraph, motoDemand)
motoClass.set_pce(0.2)
motoClass.set_vot(35)
motoClass.set_fixed_cost('toll', 0.0125)

# %%
truckDemand = proj_matrices.get_matrix(matrix_name)
truckDemand.computational_view('trucks')
truckClass = TrafficClass(truckGraph, truckDemand)
truckClass.set_pce(1.5)
truckClass.set_vot(35)
truckClass.set_fixed_cost('toll', 0.05)
# %%


assig = TrafficAssignment()
assig.set_classes([carClass, motoClass, truckClass])

assig.set_vdf("BPR")  # This is not case-sensitive # Then we set the volume delay function

assig.set_vdf_parameters({"alpha": "alpha", "beta": "beta"})  # And its parameters

assig.set_time_field("free_flow_time")
assig.set_capacity_field(f"capacity")  # The capacity and free flow travel times as they exist in the graph

# And the algorithm we want to use to assign
assig.set_algorithm('bfw')

# You would obviously pick a much tighter convergence criterium and correspondingly larger number of iterations
assig.max_iter = 30
assig.rgap_target = 0.1

assig.execute()  # we then execute the assignment
assig.save_results('our first assignment')

# %% md
# We can also plot convergence
import matplotlib.pyplot as plt

df = assig.report()
x = df.iteration.values
y = df.rgap.values

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(x, y, "blue")
plt.yscale("log")
plt.grid(True, which="both")
plt.xlabel(r"Iterations")
plt.ylabel(r"Relative Gap")
plt.show()
