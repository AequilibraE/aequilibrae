Migrating to the matrix API
===========================

The matrix API replaces the previous ``AequilibraeMatrix`` workflow with two
objects: ``AequilibraEMatrix`` for in-memory work and ``MatrixStore`` for
interacting with an OMX file on disk. The import name for the in-memory class
is now ``AequilibraEMatrix``.

Both objects provide dictionary-style access to named matrices. Prefer
``AequilibraEMatrix`` for calculations. Use ``MatrixStore`` to inspect or
update files and ``subset()`` to create an independent in-memory working set.
Use the ``from_*`` functions to construct these objects from files, arrays or
tabular data.

For general usage, see :doc:`matrix`.

Replace loading and computational views
---------------------------------------

Previously, loading a file and selecting matrices for computation were separate
operations:

.. code-block:: python

    # Previous API
    from aequilibrae.matrix import AequilibraeMatrix

    demand = AequilibraeMatrix()
    demand.load("demand.omx")
    demand.set_index("zones")
    demand.computational_view(["car", "bus"])

Use ``from_file(..., store=True)`` to open a store and create a working set
with ``subset()`` instead:

.. code-block:: python

    # Matrix API
    from aequilibrae.matrix import from_file

    with from_file("demand.omx", index_name="zones", store=True) as store:
        demand = store.subset(["car", "bus"])

    demand["car"] *= 1.1
    demand["total"] = demand["car"] + demand["bus"]

``subset()`` returns an ``AequilibraEMatrix`` containing the selected named
matrices, their zone index and file-level metadata. It remains usable after the
store closes, and changes to it do not change the source file.

There is no ``computational_view()`` or ``matrix_view`` step. Work with each
named matrix directly. Select only the matrices needed for the calculation
rather than loading the whole file into memory.

When the subset of matrices is known ahead of time, use ``from_file()`` wit the
``subset`` argument directly:

.. code-block:: python

    from aequilibrae.matrix import from_file

    demand = from_file(
        "demand.omx", index_name="zones", subset=["car", "bus"]
    )

``from_file()`` returns an in-memory object by default. Providing
``store=True`` makes it return a store instead.

Replace empty-matrix creation
-----------------------------

The previous API created named matrices and then filled their index and
values:

.. code-block:: python

    # Previous API
    import numpy as np

    matrix = AequilibraeMatrix()
    matrix.create_empty(zones=3, matrix_names=["car", "bus"])
    matrix.index[:] = [101, 205, 309]
    matrix.matrix["car"][:, :] = np.zeros((3, 3))
    matrix.matrix["bus"][:, :] = np.zeros((3, 3))

Use ``AequilibraEMatrix`` with the zone index, then add named arrays:

.. code-block:: python

    # Matrix API
    import numpy as np
    from aequilibrae.matrix import AequilibraEMatrix

    matrix = AequilibraEMatrix(index=[101, 205, 309], index_name="zones")
    matrix["car"] = np.zeros(matrix.shape)
    matrix["bus"] = np.zeros(matrix.shape)

If the arrays are already available, use ``from_dict()`` instead:

.. code-block:: python

    from aequilibrae.matrix import from_dict

    matrix = from_dict(
        {"car": np.zeros((3, 3)), "bus": np.zeros((3, 3))},
        index=[101, 205, 309],
        index_name="zones",
    )

Replace matrix access and inspection
------------------------------------

Access matrices by name rather than through ``matrix``, ``get_matrix()`` or
dynamically named attributes. Put indexes in the same brackets as the matrix
name when changing values to ensure the underly data is mutated, rather than a
copy:

.. list-table::
   :header-rows: 1

   * - Previous workflow
     - Matrix API
   * - ``matrix.get_matrix("car")``
     - ``matrix["car"]``
   * - ``matrix.get_matrix("car", copy=True)``
     - ``matrix["car"].copy()``
   * - ``matrix.matrix["car"][0, 1]``
     - ``matrix["car", 0, 1]``
   * - ``matrix.matrix["car"][:, 1] = values``
     - ``matrix["car", :, 1] = values``
   * - ``matrix.names``
     - ``list(matrix)``
   * - ``matrix.cores``
     - ``len(matrix)``
   * - ``matrix.zones``
     - ``len(matrix.index)``
   * - ``matrix.copy(cores=["car"])``
     - ``matrix.subset(["car"])``

For example:

.. code-block:: python

    matrix["car", 0, 1] = 42
    matrix["car", :, 2] = [1, 2, 3]
    selected = matrix.subset(["car"])
    print(list(selected), selected.shape)

Replace index changes
---------------------

Choose an OMX mapping with ``index_name`` when opening the file rather than
calling ``set_index()`` after loading. Each object has one selected zone index.
To use another mapping, open the file again with that mapping's name.

The retrieved index is read-only. Replace slice assignment with assignment of
the complete index:

.. code-block:: python

    # Previous API
    matrix.index[:] = [1001, 1002, 1003]

.. code-block:: python

    # Matrix API
    matrix.index = [1001, 1002, 1003]

Index values must be unique, non-negative integers. The index length cannot
change after construction. Relabelling the index does not reorder the matrix
values. Create a new object when the number of zones changes.

Replace saves and file edits
----------------------------

For in-memory results, replace ``export()`` and file-oriented ``save()`` calls
with ``save_as_omx()``:

.. code-block:: python

    # Previous API
    demand.export("adjusted_demand.omx", cores=["car", "bus"])

.. code-block:: python

    # Matrix API
    demand.save_as_omx("adjusted_demand.omx", subset=["car", "bus"])

The zone index and file-level metadata are saved with the selected matrices.
Use a new output file to preserve the source. Attempting to save over existing
matrices and index mappings will raise an exception unless ``overwrite`` is
``True``:

.. code-block:: python

    demand.save_as_omx(
        "adjusted_demand.omx", subset=["car", "bus"], overwrite=True
    )

For direct file edits, use ``from_file()`` to open a writable store:

.. code-block:: python

    with from_file(
        "demand.omx", index_name="zones", mode="r+", store=True
    ) as store:
        store["car", 0, 1] = 42
        store["bus", :, 2] = [1, 2, 3]

These assignments update the file without a separate save call.
``store["car"][:, 2] = ...`` only changes the returned array, so replace it
with ``store["car", :, 2] = ...``. The store provides file access through a
similar interface to the in-memory object, but prefer ``subset()`` and an
``AequilibraEMatrix`` for calculations.

Use a ``with`` block to close stores, or call ``store.close()`` explicitly.
In-memory objects do not need ``close()``. Use ``store.omx`` for OMX-specific
operations.

Replace descriptive attributes
------------------------------

Use ``metadata`` for file-level information instead of the old ``name`` and
``description`` attributes:

.. code-block:: python

    demand.metadata.update(
        {"name": "morning_demand", "description": "Adjusted morning demand"}
    )

Per-matrix attributes can be supplied through ``matrix_metadata`` when saving.
These attributes can then be used with ``store.subset([{"mode": "car"}])`` to
select an in-memory working set.

Replace trip-list loading
-------------------------

Replace ``create_from_trip_list()`` with a pandas read followed by
``from_df()``:

.. code-block:: python

    import pandas as pd
    from aequilibrae.matrix import from_df

    trips = pd.read_csv("trips.csv")
    demand_from_trips = from_df(
        trips, row="origin", col="destination", subset=["car", "bus"],
        index_name="zones",
    )
    demand_from_trips.save_as_omx("trip_demand.omx")

Repeated origin-destination pairs are summed, and missing pairs default to
zero. The index is inferred from the sorted unique origin and destination
IDs. Pass ``index`` explicitly when a particular zone order or extra zones
are required.

Move existing files to OMX
--------------------------

The matrix objects support OMX only for file storage. Convert existing ``.aem``
files with the previous API before moving to the new objects:

.. code-block:: python

    # Previous API, before upgrading
    legacy = AequilibraeMatrix()
    legacy.load("legacy.aem")
    legacy.export("converted.omx")
    legacy.close()
