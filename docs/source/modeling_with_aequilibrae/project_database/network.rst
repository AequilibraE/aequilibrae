.. _network:

Network
=======

The objectives of developing a network format for AequilibraE are to provide the
users a seamless integration between network data and transportation modeling
algorithms and to allow users to easily edit such networks in any GIS platform
they'd like, while ensuring consistency between network components, namely links
and nodes. As the network is composed by two tables, **links** and **nodes**,
maintaining this consistency is not a trivial task.

As mentioned in other sections of this documentation, the links and a nodes
layers are kept consistent with each other through the use of database triggers,
and the network can therefore be edited in any GIS platform or
programmatically in any fashion, as these triggers will ensure that
the two layers are kept compatible with each other by either making
other changes to the layers or preventing the changes.

**We cannot stress enough how impactful this set of spatial triggers was to**
**the transportation modeling practice, as this is the first time a**
**transportation network can be edited without specialized software that**
**requires the editing to be done inside such software.**

.. important::
   AequilibraE does not currently support turn penalties and/or bans. Their
   implementation requires a complete overahaul of the path-building code, so
   that is still a long-term goal, barred specific development efforts.

.. seealso::

   * :ref:`links_network_data_model`
      Data model
   * :ref:`nodes_network_data_model`
      Data model

.. toctree::
   :maxdepth: 1
   :caption: Dive deep into network!

   network_geometry.rst
   network_import_and_export.rst
