.. _database_migration:

Database Upgrades
=================

Occasionally AequilibraE needs to make changes to the database schemas or provide format upgrades. This changes are
delivered through a set of migration files shipped with AequilibraE.

``aequilibrae.Project.upgrade()``
--------------------------------

Database upgrades can be applied via the ``aequilibrae.Project.upgrade()`` function. All applicable upgrades will be
applied and marked as such in the ``migrations`` table of ``project_database.sqlite``. On first upgrade this table will
be created.

Database downgrades are not supported. Previous versions of AequilbraE are not guaranteed to work with newer databases.

If skipping a specific migration is required, use the ``aequilbrae.project.tools.MigrationManager`` object
directly. Consult it's documentation page for details. Take care when skipping migrations.

All database upgrades are applied within a single transaction.
