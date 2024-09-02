.. _public_transport_database:

Public Transport database
=========================

AequilibraE is capable of importing a General Transit Feed Specification (GTFS) feed 
into its database. The Transit module has been updated in version 0.9.0. More details on 
the **public_transport.sqlite** are discussed on a nearly *per-table* basis below, and 
we recommend understanding the role of each table before setting an AequilibraE model 
you intend to use. If you don't know much about GTFS, we strongly encourage you to take
a look at the documentation provided by `Google <https://developers.google.com/transit/gtfs>`_.

The public transport database is created on the run when the ``Transit`` class is executed
for the first time.