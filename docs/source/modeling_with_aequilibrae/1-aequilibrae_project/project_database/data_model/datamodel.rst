:orphan:

.. _supply_data_model:

SQL Data model
^^^^^^^^^^^^^^

The data model presented in this section pertains only to the structure of
AequilibraE's project_database and general information about the usefulness
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

   about.rst
   attributes_documentation.rst
   link_types.rst
   links.rst
   matrices.rst
   modes.rst
   nodes.rst
   periods.rst
   results.rst
   transit_graph_configs.rst
   zones.rst
