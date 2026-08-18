:orphan:

.. _database_migration:

Database upgrades
=================

Database upgrades are closed-project operations. Apply every applicable
network, results, and transit migration with the path-based API::

    Project.upgrade(project_path)

Upgrade owns one persistent named connection closure for the operation and
closes it before returning. Schema changes and migration-status writes share
each database's transaction. Because SQLite has no distributed transaction
protocol, a failure while committing several database files can leave an
earlier file committed.

Migration authoring contract
----------------------------

Migrations are Python modules that expose a ``migrate`` function accepting
``project_conn``, ``transit_conn``, and ``results_conn`` keyword arguments for
the open database connections. Migrations execute inside a transaction owned by
the migration manager. Migration functions must not call ``commit()``,
``rollback()``, ``close()``, ``transaction()``, issue transaction-control SQL,
or use a native connection context manager. They receive their database
connections from the migration runner and must not recover a project globally.

Database downgrades and selectively skipping migration databases are not
supported.
