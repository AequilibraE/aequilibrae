.. _create_your_first_model:

Your First Model
================

This is a short introduction to AequilibraE, especially prepared for new users.
You can see other applications in the :ref:`example's gallery <Examples>`.

AequilibraE currently has three built-in examples: Sioux Falls, Nauru, and Coquimbo.
Let's reproduce part of the Coquimbo model, performing traffic assignment and creating
its GTFS feed.

.. _first-model-imports:

.. code-block:: python
    
    # Imports

    from aequilibrae import Project
    from os.path import join
    from uuid import uuid4
    from tempfile import gettempdir

.. _first_model_create_project:

Create new project
------------------

.. code-block:: python
    
    folder = join(gettempdir(), uuid4().hex)

    project = Project()
    project.new(folder)

.. _first-model-create-network:

Import traffic network
----------------------

AequilibraE can create networks :ref:`from OSM <plot_from_osm>`, 
:ref:`from GMNS <import_from_gmns>`, and
:ref:`from a link layer <project_from_link_layer>`.

Coquimbo and its neighbor city La Serena form one metropolitan region.
Let's import our network from OSM using a bounding box, otherwise when we create the
GTFS feed, part of our routes will be missing.

.. code-block:: python

    project.network.create_from_osm(west=-71.402206, south=-30.042350, east=-71.134758, north=-29.853151)

.. _first-model-build-zones:

Build zones
-----------

As we're creating a model from scratch, we don't have its TAZs provided by the
transit authorities. We're going to create hex zones just like in :ref:`this example <create_zones>`.

In case you already have your own TAZs saved in shapefile or geopackage, for example, you can refer to the :ref:`importing TAZ from external files example <>`.

.. code-block:: python

.. _first-model-create-connectors:

Create centroid connectors
-------------------------

Now that we have build our zones, we can create our centroid connectors.
The centroid connectors are created by connecting the zone centroid to one or more
nodes selected from all those that satisfy *mode* and *link_type* criteria, and 
are inside the zone.

.. code-block:: python

    for zone_id, zone in zoning.all_zones().items():
        zone.connect_mode(mode_id="t", link_types="", connectors=2)

The process shall take a few minutes.

.. _first-model-build-graph:

