.. index:: project

The AequilibraE project
=======================

Similarly to commercial packages, AequilibraE is moving towards having the majority of
a model residing inside a single file. This change is motivated by the fact that it is
easier to develop and maintain documentation for models if they are kept in a format
that favors data integrity and that supports a variety of data types and uses.

The chosen format for AequilibraE is `SQLite <https://sqlite.org/index.html>`_, with
all the GIS capability supported by
`SpatiaLite <https://www.gaia-gis.it/fossil/libspatialite/index>`_. Their impressive
performance, portability, self containment and open-source character of these two
pieces of software, along with their large user base and wide industry support make
them solid options to be AequilibraE's data backend. Since working with Spatialite is
not just a matter of *pip installing* a package, please refer to :ref:`dependencies`.

.. note::
   This portion of AequilibraE is under intense development, and important changes to the
   project structure and the API should be expected until at least version 0.7. Versions
   0.8 and beyond should see a much more stable API.

One of the key characteristics of any modelling platform is the ability of the supporting
software to maintain internal data consistency. Network data consistency has been
introduced in the AequilibraE framework with  TranspoNET (transponet_), where
`Andrew O'Brien <https://www.linkedin.com/in/andrew-o-brien-5a8bb486/>`_ implemented the
first link-node consistency infrastructure in the form of spatialite triggers. Further
data consistency, especially for tabular data, is also necessary, however, and it is
being slowly introduced in the AequilibraE project in the former of database triggers and
user-triggered consistency checks through the API.

Project organization
--------------------
Along with the Sqlite file that is the base for the AequilibraE project, other data such
as documentation and binary matrices (OMX or AEM) do not fit nicely into a database
structure, and therefore are stored in the file system in parallel with the Sqlite
Project in a folder with the same name.

For now, AequilibraE only supports one network per project file, as everything related
to network names is hard-coded, and the work of re-factoring it is substantial. However,
there is nothing in the SQLite architecture that prevents housing an arbitrarily large
number of networks within the same file.

Project components
------------------

* Network
  * Links layer
  * Nodes layer
* Supporting layers
  * Zoning layer
* Matrix index
* Vectors
  * Vector index
  * Vector data
* Scenario list
* Assignment results


.. index:: transponet

Network
~~~~~~~

Transponet (Andrew's work)
Link to the documentation on Transponet


Links
+++++
A bit about it
All link data is stored in the link table itself

Nodes
+++++
 A bit about it

Supporting layers
~~~~~~~~~~~~~~~~~
Will include zone layer, but may include Delaunay triangulation in the future as well

Zone layer
++++++++++
Just for displaying purposes. No math involves this layer


Matrix Index
~~~~~~~~~~~~

Blah on the matrix index

Vectors
~~~~~~~

Blah on the vectors. Some on the index and some on the data

Scenario List
~~~~~~~~~~~~~

Due to the consistency enforcement previously mentioned, all models when created will
have a scenario called *default*, which cannot be deleted. This scenario is to be used
as the value associated to vectors ...  FINISH THIS TEXT

Summary of project tables
~~~~~~~~~~~~~~~~~~~~~~~~~

- links
- nodes
- vector_index
- vector_data
- matrix_index
- scenario_index

Project API
-----------

TODO: TALK ABOUT THE API