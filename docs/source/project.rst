.. _project:

The AequilibraE project
=======================

Similarly to commercial packages, AequilibraE is moving towards having the
majority of a model's components residing inside a single file. This change is motivated
by the fact that it is easier to develop and maintain documentation for models
if they are kept in a format that favors data integrity and that supports a
variety of data types and uses.

.. note::
  As of now, only projection WGS84, 4326 is supported in AequilibraE.
  Generalization is not guaranteed, but should come with time.

The chosen format for AequilibraE is `SQLite <https://sqlite.org/index.html>`_,
with all the GIS capability supported by
`SpatiaLite <http://www.gaia-gis.it/gaia-sins/>`_. The
impressive performance, portability, self containment and open-source character
of these two pieces of software, along with their large user base and wide
industry support make them solid options to be AequilibraE's data backend.
Since working with Spatialite is not just a matter of *pip installing* a
package, please refer to :ref:`dependencies`.

.. note::
   AequilibraE 0.7.0 brought and important changes to the project structure and
   the API. Versions 0.8 and beyond should see a much more stable API, with new
   capabilities being incorporated after that.

Project structure
-----------------
Since version 0.7, the AequilibraE project consists of a main folder, where a
series of files and sub folders exist. The files are the following:

- **project_database.sqlite** - Main project file, containing network and data
  tables for the project

- **results_database.sqlite** - Database containing outputs for all algorithms
  such as those resulting from traffic assignment

- **parameters.yml** - Contains parameters for all parameterized AequilibraE
  procedures

- **matrices** (*folder*) - Contains all matrices to be used within a project

- **scenarios** (*folder*) - Contains copies of each *project_database.sqlite*
  at the time a certain scenario was saved (upcoming in version 0.8)

Data consistency
----------------

One of the key characteristics of any modelling platform is the ability of the
supporting software to maintain internal data consistency. Network data
consistency is surely the most critical and complex aspect of overall data
consistency, which has been introduced in the AequilibraE framework with
`TranspoNET <https://www.github.com/aequilibrae/transponet>`_,  where
`Andrew O'Brien <https://www.linkedin.com/in/andrew-o-brien-5a8bb486/>`_
implemented link-node consistency infrastructure in the form of spatialite
triggers.

Further data consistency, especially for tabular data, is also necessary. This
need has been largely addressed in version 0.7, but more triggers will most
likely be added in upcoming versions.

All consistency triggers/procedures are discussed in parallel with the
features they implement.


Dealing with Geometries
-----------------------
Geometry is a key feature when dealing with transportation infrastructure and
actual travel. For this reason, all datasets in AequilibraE that correspond to
elements with physical GIS representation are geo-enabled.

This also means that the AequilibraE API needs to provide an interface to
manipulate each element's geometry in a convenient way. This is done using the
wonderful `Shapely <https://shapely.readthedocs.io/>`_, and we urge you to study
its comprehensive API before attempting to edit a feature's geometry in memory.

As AequilibraE is based on Spatialite, the user is also welcome to use its
powerful tools to manipulate your model's geometries, although that is not
recommended, as the "training wheels are off".


Project database
----------------
.. toctree::
   :maxdepth: 1

   project_docs/about
   project_docs/network
   project_docs/modes
   project_docs/link_types
   project_docs/matrices
   project_docs/zones
   project_docs/parameters_metadata
   project_docs/results

Parameters file
----------------
.. toctree::
   :maxdepth: 1

   parameter_file

Extra user data fields
~~~~~~~~~~~~~~~~~~~~~~
The AequilibraE standard configuration is not particularly minimalist, but it is
reasonable to expect that users would require further data fields in one or more
of the data tables that are part of the AequilibraE project. For this reason, and
to incentivate the creation of a consistent culture around the handling of model
data in aequilibrae, we have added 10 additional data fields to each table which
are not used by AequilibraE's standard configuration, all of which are named as
Greek letters. They are the following:

- 'alpha'
- 'beta'
- 'gamma'
- 'delta'
- 'epsilon'
- 'zeta'
- 'iota'
- 'sigma'
- 'phi'
- 'tau'