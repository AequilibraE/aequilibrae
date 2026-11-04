.. _project_backups:

Project backups
===============

Two standalone utilities copy a project without opening or activating a
``Project``. Both include the root project and every directory immediately under
``scenarios/`` containing a ``project_database.sqlite``.

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

Copy into the current schema
----------------------------

Use this to initialise databases using the installed AequilibraE version's schema
and attempt to copy the existing data into them:

.. code-block:: console

   aeq -p /models/source copy-to-new /models/rebuilt

.. code-block:: python

   from aequilibrae.project.tools import copy_to_new

   destination = copy_to_new("/models/source", "/models/rebuilt")

Only matching AequilibraE-managed tables and columns are copied. Source-only
tables and columns are omitted with warnings through Python's logging system.
This includes user-added network attributes. No migrations are run, and renamed
or moved fields are not mapped automatically. Keep a plain backup if you need to
preserve everything.

New columns use the current schema's defaults. Tables present in the source
receive source rows instead of the freshly initialised rows; tables absent from
the source retain their initial data. Migration records and SpatiaLite metadata
are freshly initialised. Spatial indexes are rebuilt while copying geometry.
AequilibraE's editing triggers are installed before any source rows are inserted,
after clearing initial rows being replaced. Tables are copied in dependency order
using both foreign keys and the dependencies of editing triggers: for example,
network modes and nodes precede links, and links precede turn restrictions.
Foreign keys remain enabled while copying the network and transit databases.
Dependency cycles cause an error rather than disabling validation.

Trigger-generated changes are accepted, including recalculated distances,
corrected link endpoints, additional nodes and regenerated turn geometry. This
validates data through the current schema's insertion behaviour; it does not
prove the source data was already consistent or preserve every source value.
Foreign-key and SQLite integrity checks also run before each database's data
transaction is committed. SpatiaLite is required.

Result tables are an exception to the fixed-schema rule: AequilibraE creates
these dynamically. Tables registered in the project's ``results`` table are
recreated in ``results_database.sqlite`` with their source definitions, columns,
data and indexes. Unregistered tables are omitted with warnings. A registered
table missing from an existing results database causes the operation to fail.
An absent results database with registered results, or a registered scenario
without its project database, also causes ``copy-to-new`` to fail before copying
accompanying files. Plain ``backup`` preserves such incomplete projects without
these additional checks.

Files and safety
----------------

Both utilities copy these databases when present, at root and scenario level:

* ``project_database.sqlite``
* ``public_transport.sqlite``
* ``results_database.sqlite``

After **all** databases have been copied successfully, they copy
``parameters.yml``, ``aequilibrae.log``, and the complete ``matrices/`` and ``run/``
folders when present. Python caches (``__pycache__``, ``*.pyc`` and ``*.pyo``) are
excluded. Other files and databases at root/scenario level are not copied.
Accompanying files are not rewritten, including any paths in ``parameters.yml``.
Symbolic links within the selected project files and folders are rejected.

The destination must not exist and must be outside the source project. Neither
utility opens a ``Project`` or writes to the source databases. Any copying error
stops the operation immediately; **partial output is left in place**, not removed
or treated as a successful backup. Choose another destination or remove the
partial output yourself before retrying.

SQLite provides consistency for each database separately, not a simultaneous
snapshot of all databases and files. Stop project writers while either utility
runs if you need a consistent whole-project snapshot. Uncommitted changes are
not copied.
