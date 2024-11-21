.. _public_transport_database:

Public Transport database
=========================

AequilibraE is capable of importing a General Transit Feed Specification (GTFS) feed 
into its database. The Transit module has been updated in version 0.9.0. More details on 
the **public_transport.sqlite** are discussed on a nearly *per-table* basis below, and 
we recommend understanding the role of each table before setting an AequilibraE model 
you intend to use. If you don't know much about GTFS, we strongly encourage you to take
a look at the documentation provided by `Mobility Data <https://gtfs.org/documentation/schedule/reference/>`_.

The public transport database is created on the run when the ``Transit`` module is executed
for the first time.

.. seealso::

    * :func:`aequilibrae.transit.Transit`
        Class documentation
    * :func:`aequilibrae.transit.TransitGraphBuilder`
        Class documentation

In the following sections, we'll dive deep into the tables existing in the public transport database.
Please notice that some tables are homonyms to the ones existing in the **project_database.sqlite**,
but its contents are related to the public transport graph building and assignment processes. 

.. toctree::
    :caption: Get to know the data structures in public transport database!
    :maxdepth: 1

    data_model/datamodel.rst