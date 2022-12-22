"""
Import GTFS
===========

On this example, we import a GTFS feed to our model. We will also perform map matching.

We use the Austin (TX) example.

"""
# %%
## Imports
from uuid import uuid4
import os
from tempfile import gettempdir
from aequilibrae.transit import Transit
from aequilibrae.project import Project
from aequilibrae.utils.create_example import create_example

"""Let's create an empty project on an arbitrary folder"""
# %%
fldr = os.path.join(gettempdir(), uuid4().hex)

prj = create_example(fldr)

# %%
data = Transit(prj)
# Now we 
transit = data.new_gtfs(agency="Capital Metro", file_path="")

"""We chose one date to import the GTFS feed into our model. 
Let's continue with 2020-04-01, but you can try other days."""
# %%
transit.load_date("2020-04-01")

"""If we want to """

"""Now we enable the map matching execution"""
# %%
transit.set_allow_map_match(True)
transit.map_match()

# %%
# Let's save our GTFS into our model
transit.save_to_disk()

#%%
prj.close()
