"""
.. _example_aequilibrae_connections:

Project Connections
===================

In this example, we show how to use AequilibraE's database connections within a project.
"""

# %%
# .. admonition:: References
#
#   * :doc:`../../aequilibrae_project`

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.project.project.Project.db_connection`
#     * :func:`aequilibrae.project.project.Project.db_connection_spatial`
#     * :func:`aequilibrae.project.project.Project.results_connection`
#     * :func:`aequilibrae.project.project.Project.transit_connection`

# %%

# Imports
from uuid import uuid4
from tempfile import gettempdir
from pathlib import Path

import geopandas as gpd
import pandas as pd

from aequilibrae.utils.create_example import create_example

# %%

# We create the example project inside our temp folder.
fldr = Path(gettempdir()) / uuid4().hex
project = create_example(fldr, "sioux_falls")

# %%
# All AequilibraE projects presents four types of connections in the form of properties:
#
#    * General connection to the project database
#    * Spatial connection to the project database
#    * General connection to the results database
#    * Spatial connection to the transit database

# %%
# Each connection can be easily accessed as follows:
with project.db_connection as conn:
    matrices = pd.read_sql("SELECT * FROM matrices", conn)

# %%
matrices

# %%
# We encourage using spatial connections only when handling spatial data.
with project.db_connection_spatial as conn:
    nodes = gpd.read_postgis("SELECT zone_id, ST_AsBinary(geometry) geom FROM zones;", con=conn, geom_col="geom", crs=4326)

# %%
nodes.head()

# %%
# For accessing both results and transit databases, the procedure is the same.
#
# When you're done, don't forget to close the project.
project.close()
