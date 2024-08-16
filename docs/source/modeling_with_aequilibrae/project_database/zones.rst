.. _tables_zones:

Zones table
===========

The default **zones** table has a **MultiPolygon** geometry type and a limited
number of fields, as most of the data is expected to be in the
**demand_database.sqlite**.

The API for manipulation of the zones table and each one of its records is
consistent with what exists to manipulate the other fields in the database.

As it happens with links and nodes, zones also have geometries associated with
them, and in this case they are of the type .

You can check :ref:`this example <create_zones>` to learn how to add zones to your project.

.. seealso::

    :func:`aequilibrae.project.Zone`