Migrating to the table API
==========================

The project-table API replaces the mutable, cached ``Link``, ``Node``,
``Zone``, ``Mode``, ``LinkType``, ``Period``, ``MatrixRecord``, and
``ResultRecord`` objects. This is a breaking change: records returned by the
new API are immutable records, and the removed classes and methods have no
compatibility aliases.

The same table workflow is used by ``links``, ``nodes``, ``modes``,
``link_types``, ``periods``, ``zones``, ``matrices``, and ``results``. The
zones table has moved from ``project.zoning`` to ``project.network.zones``.
The table class is now ``Zones`` in ``aequilibrae.project.network.zones``.

Read and inspect records
------------------------

Use ``get()`` to retrieve one record, iterate over a table to retrieve all
records, and use normal container operations to test existence or obtain the
row count::

    links = project.network.links

    if 42 in links:
        link = links.get(42)
        print(link.modes)

    print(f"The network has {len(links)} links")
    for link in links:
        print(link.link_id, link.name)

Records are frozen dataclasses. Updating a table does not alter a record that
was read earlier. Call ``get()`` again when the latest values are needed::

    link = links.get(42)
    previous_speed = link.speed_ab

    links.update(42, speed_ab=80)

    assert link.speed_ab == previous_speed
    assert links.get(42).speed_ab == 80

Generated records include fields added through the table's ``fields`` editor.
The table refreshes its generated record type after a schema change. Use
``table.columns`` to inspect writable columns and ``table.data`` when a
DataFrame or GeoDataFrame is more appropriate.

Replace mutable-record saves
----------------------------

Previously, a record was changed in memory and then persisted with ``save()``::

    # Previous API
    link = project.network.links.get(42)
    link.speed_ab = 80
    link.name = "Main Street"
    link.save()

Pass the key and changed values to ``update()`` instead::

    # Table API
    links.update(42, speed_ab=80, name="Main Street")

``update()`` changes only the supplied columns and raises ``ValueError`` if the
key does not exist. To remove a row, replace ``record.delete()`` with
``table.delete(key)``::

    project.network.nodes.delete(17)
    project.network.zones.delete(5)

``delete()`` also raises ``ValueError`` when no row has the requested key.

Create rows
-----------

Replace the staged ``new()`` and ``save()`` workflow with one ``insert()``
call. It returns the explicit or generated key::

    # Previous API
    mode = project.network.modes.new("k")
    mode.mode_name = "freight"
    project.network.modes.add(mode)
    mode.description = "Freight vehicles"
    mode.save()

    # Table API
    mode_id = project.network.modes.insert(
        mode_id="k",
        mode_name="freight",
        description="Freight vehicles",
    )
    assert mode_id == "k"

For spatial tables, pass Shapely geometry directly. Geometry is returned as a
Shapely object when the row is read::

    from shapely.geometry import LineString, Point, Polygon

    nodes = project.network.nodes
    nodes.insert(node_id=1001, is_centroid=0, geometry=Point(-43.20, -22.90))
    nodes.insert(node_id=1002, is_centroid=0, geometry=Point(-43.19, -22.90))

    link_geometry = LineString([(-43.20, -22.90), (-43.19, -22.90)])
    link_id = project.network.links.insert(
        a_node=1001,
        b_node=1002,
        modes="c",
        geometry=link_geometry,
    )

    zone_polygon = Polygon(
        [
            (-43.21, -22.91),
            (-43.18, -22.91),
            (-43.18, -22.89),
            (-43.21, -22.89),
            (-43.21, -22.91),
        ]
    )
    project.network.zones.insert(zone_id=1001, name="Downtown", geometry=zone_polygon)

For scalar insertion, ``links`` supplies defaults for omitted endpoints,
direction, and link type. Other omitted columns use their SQLite defaults or
remain ``NULL`` when the schema permits it.

``links.copy(link_id)`` replaces ``copy_link()``. The old method returned an
unsaved mutable record. The new method inserts the copy immediately and returns
its new ID::

    copied_link_id = project.network.links.copy(link_id)
    copied_link = project.network.links.get(copied_link_id)

Work with modes, link types, periods, and connectors
----------------------------------------------------

Collection methods that returned dictionaries are gone. Replace
``all_modes()``, ``all_types()``, and ``all_zones()`` with iteration::

    modes_by_id = {
        mode.mode_id: mode
        for mode in project.network.modes
    }
    link_types_by_id = {
        link_type.link_type_id: link_type
        for link_type in project.network.link_types
    }
    zones_by_id = {
        zone.zone_id: zone
        for zone in project.network.zones
    }

Use ``data`` instead when the next operation is naturally a DataFrame
selection, join, or aggregation. ``get_by_name()`` remains available for modes
and link types. Create rows with ``insert()``, change them with ``update()``,
and remove them with ``delete()``.

Operations formerly performed by a row now belong to its table:

.. list-table::
   :header-rows: 1

   * - Previous workflow
     - Table API
   * - ``node.renumber(new_id)``
     - ``nodes.renumber(node_id, new_id)``
   * - ``period.renumber(new_id)``
     - ``periods.renumber(period_id, new_id)``
   * - ``link.set_modes("cb")``
     - ``links.update(link_id, modes="cb")``
   * - ``link.add_mode("b")``
     - ``links.add_mode(link_id, "b")``
   * - ``link.drop_mode("b")``
     - ``links.drop_mode(link_id, "b")``
   * - ``zone.add_centroid(point)``
     - ``zones.add_centroid(zone_id, point)``
   * - ``zone.disconnect_mode("c")``
     - ``zones.disconnect_mode("c", zone_id=zone_id)``
   * - ``node.connect_mode("c")``
     - ``nodes.connect_mode(node_id, "c")``
   * - ``zoning.create_zoning_layer()``
     - ``zones.create_zones_table()``
   * - ``zoning.has_zoning``
     - ``zones.has_zones``

Connecting one zone requires its geometry to limit the node search, just as
the previous row method did::

    zone_id = 1001
    zone = project.network.zones.get(zone_id)
    project.network.nodes.connect_mode(
        zone_id,
        "c",
        connectors=2,
        area=zone.geometry,
    )

To connect every zone that has a centroid, use the table-wide operation::

    project.network.zones.connect_mode("c", connectors=2, limit_to_zone=True)

``nodes.new_centroid(node_id, geometry)`` remains as a convenience method, but
it now requires the geometry and inserts the centroid immediately::

    # Previous API
    centroid = project.network.nodes.new_centroid(1001)
    centroid.geometry = Point(-43.195, -22.900)
    centroid.save()

    # Table API
    project.network.nodes.new_centroid(1001, Point(-43.195, -22.900))

``periods.new_period(...)`` likewise inserts immediately and returns the period
ID::

    project.network.periods.new_period(
        2,
        start=7 * 60 * 60,
        end=9 * 60 * 60,
        description="Morning peak",
    )

Load or update many rows
------------------------

Use ``insert_from()`` and ``update_from()`` for DataFrame workflows. Both
operations are atomic. ``insert_from()`` accepts a DataFrame with an explicit
key column and returns the inserted keys. For numeric-key tables, omitting the
key column allocates sequential IDs. For non-numeric-keys, the keys must be
provided::

    import pandas as pd

    new_links = pd.DataFrame(
        {
            "a_node": [1001, 1002],
            "b_node": [1002, 1003],
            "direction": [0, 0],
            "modes": ["c", "cb"],
            "link_type": ["default", "default"],
            "geometry": [first_geometry.wkb, second_geometry.wkb],
        }
    )
    new_ids = project.network.links.insert_from(new_links)

Bulk insertion uses the columns in the DataFrame and any defaults. It does
not apply the scalar ``insert()`` conversions. Values must therefore already
be serialised for SQLite. In particular, spatial geometry values must be WKB
rather than Shapely objects.

``update_from()`` requires the table key as a DataFrame column. Other columns
are the values to write. If any key does not exist, the operation raises
``ValueError`` before changing the table. The error reports at most the first
10 missing keys. The return value is the number of submitted rows::

    changes = pd.DataFrame(
        {
            "link_id": [new_ids[0], new_ids[1]],
            "speed_ab": [50.0, 40.0],
            "capacity_ab": [1800.0, 1200.0],
        }
    )
    submitted = project.network.links.update_from(changes)
    assert submitted == 2

Neither method mutates the caller's DataFrame. Use scalar ``insert()`` or
``update()`` when each row needs separate error handling.

Transactions and custom table integrations
-------------------------------------------

A standalone scalar operation opens and finalises a transaction automatically.
Inside an existing transaction, scalar mutations join that transaction. Bulk
operations always open their own nested transaction scope.

Nested transaction scopes use SQLite savepoints. If an exception from an inner
scope is caught by an outer scope, only the inner scope is rolled back.

Matrices and results
--------------------

Matrix and result records support the same ``get()``, metadata ``update()``,
and metadata ``delete()`` operations. Use their file- and table-aware helpers
when the associated matrix file or result table must change too.

For matrices, replace ``new_record()`` with ``create()`` to export an in-memory
matrix and register it. Exporting through this method supports OMX files::

    matrix_record = project.matrices.create(
        "demand_2030",
        "demand_2030.omx",
        matrix=demand,
    )
    print(matrix_record.file_name, matrix_record.cores)

Use ``register_matrix()`` when the file already exists in the project's matrix
directory::

    skim_record = project.matrices.register_matrix(
        "base_skims",
        "base_skims.omx",
    )
    base_skims = project.matrices.get_matrix(skim_record.name)

Metadata changes remain table operations::

    project.matrices.update("demand_2030", description="2030 demand")
    project.matrices.delete_matrix("demand_2030")

``delete_matrix()`` removes both metadata and the matrix file. In contrast,
``project.matrices.delete(name)`` removes only metadata and leaves the file in
place.

For results, replace ``new_record()`` followed by ``set_data()`` with one
``create()`` call. Named DataFrame index levels are stored as regular columns,
which makes a link-indexed result straightforward to retrieve::

    assignment_data = pd.DataFrame(
        {
            "volume_ab": [1250.0, 830.0],
            "volume_ba": [1100.0, 790.0],
        },
        index=pd.Index([42, 43], name="link_id"),
    )

    result = project.results.create(
        "assignment_2030",
        assignment_data,
        procedure="traffic assignment",
        procedure_id="2030-base",
        procedure_report={"converged": True},
        year="2030",
    )

    stored_data = project.results.get_results(result.table_name)
    assert set(stored_data["link_id"]) == {42, 43}

    project.results.delete_result(result.table_name)

``create()`` raises an error rather than replacing an existing metadata record
or data table. ``delete_result()`` removes both metadata and its data table.
Generic ``results.delete(name)`` removes metadata only.

Both ``matrices`` and ``results`` provide ``clear_database()`` to remove stale
metadata, ``update_database()`` to register unrecorded resources, and ``sync()``
to run both reconciliation steps.
