AequilibraE 2.0 project API migration
=====================================

AequilibraE 2.0 makes project and transaction ownership explicit. This is a
clean cutover; the removed APIs have no compatibility aliases.

Project lifetime and ownership
------------------------------

Do not depend on a process-wide active project, and do not call ``close()``::

    # 1.x
    project = Project.from_path(path)
    do_work_that_finds_the_active_project()
    project.close()

    # 2.0
    with Project.from_path(path) as project:
        do_work(project)

Hosts that cannot use a context manager call the idempotent
``project.shutdown()``. A selected ``Scenario`` owns its paths, gateways, and
persistent project, results, and transit connections. Several projects may be
open concurrently.

Transactions and low-level SQL
------------------------------

Standalone gateway writes commit before returning::

    project.network.links.update(42, speed_ab=80)

Coordinate gateway writes explicitly when required::

    with project.transaction():
        project.network.links.update(42, speed_ab=80)
        project.network.nodes.update(7, is_centroid=1)
        assert project.network.links.get(42).speed_ab == 80

The context binds ``None``. Nested project transactions create a savepoint on
each named scenario connection. An exception rolls back all connections whose
transaction remains active. SQLite cannot atomically commit independent files,
so a failure during multi-connection commit can leave an earlier file
committed.

``project.db_connection`` is the persistent project-database transaction
manager, not a connection context factory::

    rows = project.db_connection.execute("SELECT link_id FROM links").fetchall()
    with project.db_connection.transaction():
        project.db_connection.execute("UPDATE links SET speed_ab=? WHERE link_id=?", (80, 42))

It exposes no ``commit()``, ``rollback()``, ``close()``, or raw connection.
``database_connection()``, ``db_connection_spatial``, ``results_connection``,
and ``transit_connection`` were removed.

Table writes and immutable records
----------------------------------

``get()`` returns an immutable snapshot. Replace mutable records and ``save``::

    # 1.x
    link = links.get(42)
    link.speed_ab = 80
    link.save()

    # 2.0
    links.update(42, speed_ab=80)

``new_record()`` and ``set_data()`` were removed; use ``insert()`` and the bulk
methods. Public ``conn=`` parameters, ``TableBatch``, and ``batch()`` were also
removed::

    links.insert(a_node=1, b_node=2, modes="c")
    links.insert_from(new_links)
    links.update_from(changed_links)

Every scalar or DataFrame mutation is automatically atomic. Inside a project
transaction, each gateway operation uses a savepoint, so a caught bulk-write
failure cannot leave its successful prefix pending.

DataFrame row identity
----------------------

``table.data`` is indexed by the table key and does not duplicate that key as a
column::

    frame = project.network.links.data
    assert frame.index.name == "link_id"
    frame.loc[42, "speed_ab"] = 80
    project.network.links.update_from(frame[["speed_ab"]])

``update_from`` requires a uniquely valued, non-missing index whose name is the
table key. A key value column is rejected. The caller's index is not reset or
mutated. Use ``reset_index()`` explicitly only at boundaries requiring a key
column, such as an export or column-oriented merge. ``insert_from`` accepts
ordinary value columns and performs one atomic insertion operation.

Matrices and results
--------------------

Inherited CRUD now changes metadata records only::

    project.matrices.update("demand", description="Base demand")
    project.matrices.delete("demand")       # file remains
    project.results.delete("assignment")   # payload table remains

Resource-aware operations are explicit::

    project.matrices.create(...)
    project.matrices.delete_matrix("demand")
    project.results.create("assignment", dataframe, procedure="assignment")
    project.results.delete_result("assignment")

Matrix helpers compensate file and metadata failures. Result creation always
fails if either metadata or a payload table exists; ``if_exists`` and arbitrary
``DataFrame.to_sql`` options are not accepted. It stores every DataFrame index
level as payload columns, writes bounded chunks in one results transaction, and
then creates metadata with compensation. Resource helpers are rejected inside
``project.transaction()`` because filesystem and independent SQLite resources
cannot participate atomically.
