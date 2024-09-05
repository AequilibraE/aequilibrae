.. _accessing_project_data:

Accessing Project Data
======================

An AequilibraE project helds geometric information that can be accessed by the user in 
three different classes: ``Links``, ``Nodes``, and ``Zoning``. In this section, we'll
cover the main points regarding them.

``Links``
---------

The ``Links`` class allows the access to the links table in the project network, if one
wants to manipulate it.

.. code-block:: python

    from tempfile import gettempdir
    from aequilibrae.utils.create_example import create_example

    # Let's use Coquimbo as example
    project = create_example(gettempdir(), "coquimbo")

    links = project.network.links  # access the links table

The actions 

.. seealso::
    
    * :func:`aequilibrae.network.Links`
        Class documentation

    * :ref:`project_from_link_layer`
        Usage example

``Nodes``
---------

.. seealso::

    * :func:`aequilibrae.network.Nodes`

.. _project_zoning:

``Zoning``
----------

.. seealso::

    * :func:`aequilibrae.network.Zoning`
