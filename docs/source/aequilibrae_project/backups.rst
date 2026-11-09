.. _project_backups:

Project backups
===============

Two standalone utilities back up or import project data without opening or
activating a ``Project``. Plain backups include the root project and every directory
immediately under ``scenarios/`` containing a ``project_database.sqlite``. Imports
currently handle only the root project's ``project_database.sqlite``.

SQLite backup
-------------

Use this to preserve the existing database schemas and all their data, including
custom tables, columns, indexes and triggers:

.. code-block:: console

   aeq -p /models/source backup /backups/source

.. code-block:: python

   from aequilibrae.project.tools import backup

   destination = backup("/models/source", "/backups/source")

The databases are copied using SQLite's built-in online backup API, not by copying
open database files. Committed data in a write-ahead log is included. The result is
a consistent database copy, not necessarily a byte-for-byte identical file.
SpatiaLite is not required for this operation.

A ``tqdm`` progress bar counts completed databases across the root project and its
scenarios, showing the current database's relative path. Missing databases and
accompanying files do not count towards the total. Set ``AEQ_SHOW_PROGRESS=FALSE``
to hide the progress bar.

Import into an existing schema
------------------------------

Create a blank project separately, optionally add your own tables and columns,
and then import rows from an old project's network database:

.. code-block:: console

   aeq -p /models/rebuilt create
   aeq -p /models/rebuilt copy-from-old /models/source

.. code-block:: python

   from aequilibrae import Project
   from aequilibrae.project.tools import copy_from_old

   project = Project()
   project.new("/models/rebuilt")
   project.close()
   # Add any custom destination schema before importing.
   destination = copy_from_old("/models/source", "/models/rebuilt")

Both project databases must already exist. The actual destination schema, including
user-added tables and columns, determines what is copied; the import does not
create or alter table definitions. Source-only tables and columns are omitted.
Tables absent from the source remain untouched. Python's logging system reports
which tables and columns will be copied, omitted or supplied with defaults or NULL.
A ``tqdm`` progress bar shows completed tables and the current table's name. Only
tables being copied count towards the total. Set ``AEQ_SHOW_PROGRESS=FALSE`` to
hide the progress bar.

For every writable destination column absent from a matching source table, its
destination default is used, or NULL if no default exists. Destination constraints
still apply: for example, inserting NULL into a NOT NULL column without a default
aborts the import. Empty source tables insert no rows. Generated columns are
calculated by the destination. No migrations are run, and renamed or moved fields are not mapped
automatically. Keep a plain backup if you need to preserve everything.

SQLite and SpatiaLite infrastructure and migration records remain those of the
destination. Spatial indexes are maintained while copying geometry. Rows are
copied within SQLite using a read-only attached source database, not transferred
through Python. Tables are copied in dependency order using both foreign keys and
the dependencies of editing triggers: for example, network modes and nodes precede
links, and links precede turn restrictions. Custom foreign-key dependencies are
included. Dependency cycles cause an error rather than disabling validation.

``INSERT OR REPLACE`` replaces conflicting initial rows. Initial rows without a
source conflict remain. Use a freshly initialised destination, not a populated
project; this restriction is not yet checked. Foreign keys and insertion triggers
remain enabled. Recursive triggers are disabled so implicit replacement deletions
do not run deletion-protection triggers. Replacement is a deletion followed by an
insertion, not an update, and can affect dependent rows in a populated database.

Trigger-generated changes are accepted, including recalculated distances,
corrected link endpoints, additional nodes and regenerated turn geometry. This
validates data through the destination schema's insertion behaviour; it does not
prove the source data was already consistent or preserve every source value.
Foreign-key and SQLite integrity checks run before the import transaction is
committed. An error rolls back all imported rows. SpatiaLite is required for a
spatial destination.

Only the root ``project_database.sqlite`` is imported. Transit, results and scenario
databases and accompanying files are untouched, and references to those resources
are not checked. In particular, copying their registration rows does not copy the
registered databases, result tables or matrix files.

Files and safety
----------------

Plain backups copy these databases when present, at root and scenario level:

* ``project_database.sqlite``
* ``public_transport.sqlite``
* ``results_database.sqlite``

After **all** databases have been copied successfully, they copy
``parameters.yml``, ``aequilibrae.log``, and the complete ``matrices/`` and ``run/``
folders when present. Python caches (``__pycache__``, ``*.pyc`` and ``*.pyo``) are
excluded. Other files and databases at root/scenario level are not copied.
Accompanying files are not rewritten, including any paths in ``parameters.yml``.
Symbolic links within the selected project files and folders are rejected.

For plain backups, the destination must not exist and must be outside the source
project. Any backup error stops the operation immediately; **partial output is
left in place**, not removed or treated as a successful backup. Choose another
destination or remove the partial output yourself before retrying.

For imports, the destination project database must already exist and must not be
the source database. Neither utility opens a ``Project`` or writes to the source
databases. Symbolic links to selected database files are rejected.

SQLite provides consistency for each database separately, not a simultaneous
snapshot of all databases and files. Stop project writers while either utility
runs if you need a consistent whole-project snapshot. Uncommitted changes are
not copied.
