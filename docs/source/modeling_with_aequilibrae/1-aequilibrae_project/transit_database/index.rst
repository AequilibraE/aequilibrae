.. _public_transport_database:

Public Transport database
=========================

AequilibraE is capable of importing a General Transit Feed Specification (GTFS) feed 
into its database. The Transit module has been updated in version 0.9.0. More details on 
the **public_transport.sqlite** are discussed on a nearly *per-table* basis below, and 
we recommend understanding the role of each table before setting an AequilibraE model 
you intend to use. If you don't know much about GTFS, we strongly encourage you to take
a look at the documentation provided by `Google <https://developers.google.com/transit/gtfs>`_.

The public transport database is created on the run when the ``Transit`` module is executed
for the first time.

.. Só é possível popular essas tabelas através da execução dos módulos ``Transit`` e 
.. ``TransitGraphBuilder`` (mais algum módulo pertinente). A tabela a seguir mostra
.. quais tabelas são povoadas através dos módulos 

.. seealso::

    :func:`aequilibrae.transit.Transit`

    :func:`aequilibrae.transit.TransitGraphBuilder`

In the following sections, we'll dive deep into the tables existing in the public transport database.
Please notice that some tables are homonyms to the ones existing in the **project_database.sqlite**,
but its contents are related to the public transport graph building and assignment processes. 

.. include:: data_model/agencies.rst

.. include:: data_model/attributes_documentation.rst

.. include:: data_model/fare_attributes.rst

.. include:: data_model/fare_rules.rst

.. include:: data_model/fare_zones.rst

.. include:: data_model/link_types.rst

.. include:: data_model/links.rst

.. include:: data_model/modes.rst

.. include:: data_model/node_types.rst

.. include:: data_model/nodes.rst

.. include:: data_model/pattern_mapping.rst

.. include:: data_model/results.rst

.. include:: data_model/route_links.rst

.. include:: data_model/routes.rst

.. include:: data_model/stop_connectors.rst

.. include:: data_model/stops.rst

.. include:: data_model/trigger_settings.rst

.. include:: data_model/trips_schedule.rst

.. include:: data_model/trips.rst
