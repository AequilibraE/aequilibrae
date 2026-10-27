:orphan:

.. _database_migration:

Database Upgrades
=================

Occasionally AequilibraE needs to make changes to the database schemas or provide format upgrades. These changes are
delivered through a set of migration files shipped with AequilibraE.

.. code-block:: python

    project.upgrade()

Database upgrades can be applied via the ``project.upgrade()`` function. All applicable upgrades will be applied and
marked as such in the ``migrations`` table of ``project_database.sqlite``. On first upgrade this table will be created.
The upgrade applies to the root scenario and to every other scenario of the project, each of which has its own
databases.

Database downgrades are not supported. Previous versions of AequilibraE are not guaranteed to work with newer databases.
Copy the project folder before upgrading: the copy is the only way back.

Some upgrades drop data that can be rebuilt. Upgrading to AequilibraE 2.0 drops saved transit graphs, and logs the
configuration of each one so it can be built again (see :doc:`transit_api_migration`).

If skipping a specific migration is required, use the ``aequilibrae.project.tools.MigrationManager`` object
directly. Consult its documentation page for details. Take care when skipping migrations.

All database upgrades are applied within a single transaction.
