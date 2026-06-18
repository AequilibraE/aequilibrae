"""
.. _plot_from_overture:

Create project from Overture Maps
=================================

In this example we show how to create an empty project and populate it with a
network from Overture Maps using ``Network.import_from_overture``.

Install the optional dependencies first::

    pip install aequilibrae[create]
"""
# %%
# .. admonition:: References
#
#   * :ref:`importing_from_overture`

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.project.network.network.Network.import_from_overture`

# %%
from os.path import join
from tempfile import gettempdir
from uuid import uuid4

from shapely.geometry import box

from aequilibrae import Project

# %%
# Create an empty project on an arbitrary folder.
fldr = join(gettempdir(), uuid4().hex)

project = Project()
project.new(fldr)

# %%
# Overture imports require an EPSG:4326 polygon. The importer always uses the latest release advertised by
# Overture's STAC catalog and records that release in the project's ``about`` table.
model_area = box(-112.185, 36.59, -112.179, 36.60)
project.network.import_from_overture(model_area=model_area, modes=("car", "walk"), simplify=False)

# %%
links = project.network.links.data
links.explore(color="blue", style_kwds={"weight": 2}, tooltip="link_type")

# %%
project.close()
