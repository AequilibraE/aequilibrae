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

    project = create_example("/path_to_my_folder", "coquimbo")

    project_nodes = project.network.nodes

    # To add special generators, we can add a `new_centroid`

.. seealso::

    * :func:`aequilibrae.network.Nodes`

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
