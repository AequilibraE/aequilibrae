.. _accessing_project_data:

Accessing Project Data
======================

An AequilibraE project helds geometric information that can be accessed by the user in 
three different classes: ``Links``, ``Nodes``, and ``Zoning``. In this section, we'll
cover the main points regarding them.

``project.network.links``
-------------------------

This method allows you to access the API resources to manipulate the 'links' table.
Each item in the 'links' table is a ``Link`` object.

.. code-block:: python

    from aequilibrae.utils.create_example import create_example

    project = create_example("/path_to_my_folder", "coquimbo")

    links = project.network.links

.. seealso::
    
    * :func:`aequilibrae.network.Links`
        Class documentation

    * :ref:`project_from_link_layer`
        Usage example

``project.network.nodes``
-------------------------

This method allows you to access the API resources to manipulate the 'nodes' table.
Each item in the 'nodes' table is a ``Node`` object.

.. code-block:: python

    from aequilibrae.utils.create_example import create_example
    from shapely.geometry import Point

    project = create_example("/path_to_my_folder", "coquimbo")

    project_nodes = project.network.nodes

    # To get one 'Node' object
    node = project_nodes.get(10070)

    # We can check the existing fields for each node in the 'nodes' table
    node.data_fields()

    # Let's renumber this node, and save it
    node.renumber(1000)
    node.save()

    # A node can also be used to add a special generator
    centroid = project_nodes.new_centroid(2000)

    # Don't forget to add a geometry to your centroid if it's a new node
    centroid.geometry = Point(-71.32, -29.94)

    # As this centroid is not associated with a zone, we must tell AequilibraE the initial area around
    # the centroid to look for candidate nodes to which the centroid can connect.
    centroid.connect_mode(area=centroid.geometry.buffer(0.01), mode_id="c", connectors=1)

    # Let's save our centroid connector
    project_nodes.save()

    # And don't forget to update these changes to the nodes in memory
    project_nodes.refresh()

    # Lastly but not less important, you can check your project nodes
    # `project_nodes.data` returns a geopandas GeoDataFrame.
    project_nodes.data

    # or if you want to check the coordinate of each node in the shape of
    # a Pandas DataFrame
    project_nodes.lonlat

.. seealso::

    * :func:`aequilibrae.network.Nodes`
        Class documentation

.. _project_zoning:

``project.zoning``
------------------

This method allows you to access the API resources to manipulate the 'zones' table.
Each item in the 'zones' table is a ``Zone`` object.

.. code-block:: python

    from aequilibrae.utils.create_example import create_example

    project = create_example("/path_to_my_folder", "coquimbo")

    zones = project.zoning

.. seealso::

    * :func:`aequilibrae.network.Zoning`
        Class documentation
