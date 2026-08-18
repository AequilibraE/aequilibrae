:orphan:

.. _database_migration:

Database upgrades
=================

Database upgrades are closed-project operations. Apply every applicable
network, results, and transit migration with the path-based API::

    Project.upgrade(project_path)

The project path must not be open in this or another process. Upgrade owns one
persistent named connection closure for the operation and closes it before
returning. Schema changes and migration-status writes share each database's
transaction. Because SQLite has no distributed transaction protocol, a failure
while committing several database files can leave an earlier file committed.

Migration authoring contract
----------------------------

Both Python and SQL migrations execute inside a transaction owned by the
migration manager. They must not call ``commit()``, ``rollback()``, ``close()``,
``transaction()``, issue transaction-control SQL, or use a native connection
context manager. Python migrations receive the named ``ConnectionClosure`` and
execute through its managers without finalizing them.

``sqlite3.Connection.executescript()`` is prohibited for both Python and SQL
migrations because it implicitly commits. SQL migration files must terminate
every statement with a semicolon. The migration parser uses
``sqlite3.complete_statement()``, so trigger bodies may contain internal
semicolons, but an unterminated trailing statement is rejected.

Database downgrades and selectively skipping migration databases are not
supported.
