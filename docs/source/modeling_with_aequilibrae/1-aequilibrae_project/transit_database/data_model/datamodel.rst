:orphan:

.. _transit_supply_data_model:

SQL Data model
^^^^^^^^^^^^^^

The data model presented in this section pertains only to the structure of
AequilibraE's public_transport database and general information about the usefulness
of specific fields, especially on the interdependency between tables.

Conventions
'''''''''''

A few conventions have been adopted in the definition of the data model and some
are listed below:

- Geometry field is always called **geometry**
- Projection is 4326 (WGS84)
- Tables are all in all lower case


.. Do not touch below this line unless you know EXACTLY what you are doing.
.. it will be automatically populated

Project tables
''''''''''''''

.. toctree::
   :maxdepth: 1

   agencies.rst
   attributes_documentation.rst
   fare_attributes.rst
   fare_rules.rst
   fare_zones.rst
   link_types.rst
   links.rst
   modes.rst
   node_types.rst
   nodes.rst
   pattern_mapping.rst
   results.rst
   route_links.rst
   routes.rst
   stop_connectors.rst
   stops.rst
   trigger_settings.rst
   trips.rst
   trips_schedule.rst
