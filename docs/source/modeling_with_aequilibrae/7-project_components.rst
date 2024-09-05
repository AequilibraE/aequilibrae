Project Components
==================

In the :ref:`aeq_project_structure` section, we presented the main file components and folders that
consists an AequilibraE project. We also present in :ref:`aeq_project_database_tables` all tables
that are part of the project database, how do they look like, and what fields do they have.

The components of an AequilibraE project are:

* ``project.About``
* ``project.FieldEditor``
* ``project.Log``
* ``project.Matrices``
* ``project.Network``
* ``project.Zoning``

Network and Zoning are the components that contain the geo-spatial information of the project, such
as links, nodes, and zones, which can also be manipulated. In the Network component, there are also
non-geometric classes related to the project network, such as Modes, LinkTypes, and Periods.

One important thing to observe is that related to each component in Network and Zoning, there is an
object with similar name that corresponds to one object in the class. Thus the ``project.network.Links``
enables the access to manipulate the links table, and each item in the items table is a 
``project.network.Link``.

.. image:: ../images/project_components_and_items.png
   :align: center
   :alt: basics on project components

In this section, we'll briefly discuss about the project components without geo-spatial information.

``project.About``
-----------------

This class provides an interface for editing the 'about' table of a project. We can add new fields or
edit the existing ones as necessary, but everytime you add or modify a field, you have to write back
this information, otherwise it will be lost.

.. code-block::  python

    from aequilibrae.utils.create_example import create_example

    project = create_example("/path_to_my_folder", "coquimbo")

    project.about.add_info_field("my_new_field")
    project.about.my_new_field = "add some useful information about the field"
    
    # We can add data to an existing field
    project.about.author = "Your Name" 

    # And save our modifications
    project.about.write_back()

To check if ``my_new_field`` was added to the 'about' table, we can check all the characteristics stored
in the table.

.. code-block:: python

    project.about.list_fields()  # returns a list with all charactetistics in the 'about' table

The 'about' table is created automatically when a project is created, but if you're loading a project
created with an older AequilibraE version that didn't contain it, it is possible to create one too.

.. code-block:: python

    project.about.create()
    # All AequilibraE's example already have an 'about' table, so you don't have to create it

.. seealso::

    :ref:`tables_about`

``project.FieldEditor``
-----------------------

The ``FieldEditor`` allows the user to edit the project data tables, and it has two different purposes:

* Managing data tables, through the addition/deletion of fields
* Editing the tables' metadata (aka the description of each field)

This class is directly accessed from within the corresponding module one wants to edit.

.. code-block:: python

    from aequilibrae.utils.create_example import create_example

    project = create_example("/path_to_my_folder", "coquimbo")

    link_fields = project.network.links.fields
    # To add a new field to the 'links' table
    link_fields.add("my_new_field", "this is an example of AequilibraE's funcionalities", "TEXT")

    # Don't forget to save these modifications
    link_fields.save()

    # To edit the description of a field
    link_fields.osm_id = "number of the osm link_id"

    # Or just to access the description of a field
    link_fields.a_node

One can also check all the fields in the links table.

.. code-block:: python

    link_fields.all_fields()

All field descriptions are kept in the table 'attributes_documentation'.

.. seealso::

    :ref:`parameters_metadata`

``project.Log``
---------------

``project.Matrices``
--------------------

``project.network.LinkTypes``
-----------------------------

``project.network.Modes``
-------------------------

``project.network.Periods``
---------------------------
