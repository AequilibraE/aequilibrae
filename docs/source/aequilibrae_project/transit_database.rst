.. _public_transport_database:

Public Transport Database
=========================

AequilibraE's transit module has been updated in version 0.9.0 and more details on 
the **public_transport.sqlite** are discussed on a nearly *per-table* basis below. We
recommend understanding the role of each table before setting an AequilibraE model 
you intend to use.

The public transport database is created on the run when the ``Transit`` module is executed
for the first time and it can take a little while. 

.. seealso::

    * :func:`aequilibrae.transit.Transit`
        Class documentation
    * :func:`aequilibrae.transit.TransitGraphBuilder`
        Class documentation

In the following sections, we'll dive deep into the tables existing in the public transport database.
Please notice that some tables are homonyms to the ones existing in the **project_database.sqlite**,
but its contents are related to the public transport graph building and assignment processes. 
