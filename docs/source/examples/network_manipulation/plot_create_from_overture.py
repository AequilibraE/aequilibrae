"""
.. _plot_from_overture:
.. _plot_from_osm:

Create project from Overture Maps (or OpenStreetMap)
====================================================

In this example we show how to create an empty project and populate it with a
network. Overture Maps is the recommended default source; importing from
OpenStreetMap is also supported and is shown as commented-out code that you can
switch to.

Install the optional dependencies first::

    pip install aequilibrae[create]
"""
# %%
# .. admonition:: References
#
#   * :ref:`importing_from_overture`
#   * :ref:`importing_from_osm`

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.project.network.network.Importer.overture`
#     * :func:`aequilibrae.project.network.network.Importer.osm`

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
# Both Overture and OSM imports require an EPSG:4326 polygon.
model_area = box(-112.185, 36.59, -112.179, 36.60)

# %%
# Import from Overture Maps (the recommended default). The importer always uses
# the latest release advertised by Overture's STAC catalog and records that
# release in the project's ``about`` table.
project.network.importer.overture(model_area=model_area, modes=("car", "walk"), simplify=False)

# %%
# To import from OpenStreetMap instead, comment out the Overture call above and
# uncomment the block below. OSM data is fetched from an Overpass endpoint; for
# large areas you should point at your own/regional Overpass server and increase
# the request timeout (the "sleep"/wait budget for Overpass) to avoid throttling.
# Both are set on the project parameters before importing:
#
# .. code-block:: python
#
#     params = project.project_parameters
#     params.parameters["osm"]["overpass_endpoint"] = "https://overpass-api.de/api"  # your own/regional server
#     params.parameters["osm"]["timeout"] = 180  # seconds to wait for Overpass before giving up
#     params.write_back()
#
#     project.network.importer.osm(model_area=model_area, modes=("car", "walk"), simplify=False)
#     # or import a whole place by name:
#     # project.network.importer.osm(place_name="Nauru")
#     # or import from a local extract:
#     # project.network.importer.osm(pbf_path="path/to/extract.osm.pbf")

# %%
links = project.network.links.data
links.explore(color="blue", style_kwds={"weight": 2}, tooltip="link_type")

# %%
project.close()
