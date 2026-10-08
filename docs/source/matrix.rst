Working with matrices
=====================

AequilibraE provides two objects for working with named matrices:

* ``AequilibraEMatrix`` holds matrices in memory and is backed by a dictionary.
  Prefer this object for calculations and other in-memory work.
* ``MatrixStore`` is backed by an OMX file. Use it to read and update matrices
  on disk through a similar interface to the in-memory object.

Both objects associate names with square matrices. All matrices in an object
share the same zone index, which gives the zone IDs in row and column order.
Use the ``from_file()``, ``from_dict()``, ``from_df()`` and ``from_scipy()``
functions to create the appropriate object from your source data, or the
``AequilibraEMatrix`` constructor directly.

These objects replace the previous ``AequilibraEMatrix`` which mixed both
roles. For a migration guide, see :doc:`matrix_api_migration`.

``AequilibraEMatrix`` and ``MatrixStore`` both implement the ``MutableMapping``
protocol and behave very similarly to a dictionary. You make use all mapping
operations including ``.keys()``, ``.values()``, ``.items()``, ``.update()``
and others.

Create an in-memory working set
-------------------------------

The recommended workflow is to open an OMX file with ``from_file(...,
subset=[...])``, which returns an ``AequilibraEMatrix``. If you don't know
which matrices you need ahead of time, provide ``store=True`` to ``from_file``
instead of ``subset`` to obtain a ``MatrixStore``. Inspect the file, then use
its ``subset()`` method to select the matrices needed for a calculation. It
returns an new, independent ``AequilibraEMatrix``. Work on that object in memory,
then save the result when needed.

The following example assumes that ``demand.omx`` contains matrices named
``car`` and ``bus`` and a zone mapping named ``zones``:

.. code-block:: python

    from aequilibrae.matrix import from_file

    working = from_file("demand.omx", index_name="zones", subset=["car", "bus"])
    working["car"] *= 1.1
    working["total"] = working["car"] + working["bus"]
    working.save_as_omx("working_demand.omx")

.. code-block:: python

    from aequilibrae.matrix import from_file

    with from_file("demand.omx", index_name="zones", store=True) as store:
        print(list(store))  # Prints all matrices, include "car" and "bus"
        working = store.subset(["car", "bus"])

    working["car"] *= 1.1
    working["total"] = working["car"] + working["bus"]
    working.save_as_omx("working_demand.omx")

The working set remains available after the store closes. Changes to it do not
change the source file. ``subset()`` copies the zone index and file-level
metadata as well as the specified matrices.

Use ``MatrixStore`` for manipulating files rather than working with it as if it
was in-memory. It reads the matrix from disk each time it is accessed. For this
reason, arrays obtained from it are copies, mutating them without using fancy
indexing (see ``Read and change values`` below) will mutate the copy rather
than the file itself.

Create matrices in memory
-------------------------

Use ``from_dict()`` when arrays are already available:

.. code-block:: python

    import numpy as np
    from aequilibrae.matrix import AequilibraEMatrix, from_dict

    index = [101, 205, 309]
    car = np.array([[0, 12, 8], [10, 0, 6], [7, 9, 0]], dtype=float)
    bus = np.array([[0, 2, 1], [2, 0, 3], [4, 5, 0]], dtype=float)

    demand = from_dict(
        {"car": car, "bus": bus},
        index=index,
        index_name="zones",
        metadata={"description": "Morning demand", "year": 2030},
    )

Alternatively, start with no named matrices and add them using ``AequilibraEMatrix`` directly:

.. code-block:: python

    empty = AequilibraEMatrix(index=index, index_name="zones")
    empty["car"] = np.zeros(empty.shape)
    empty.update({"bus": bus})

Matrix assignment requires a NumPy array with the object's square shape.
``AequilibraEMatrix`` copies supplied arrays and stores their values as
``float64``. Changing an array supplied earlier does not change the matrix.

Inspect and manage named matrices
---------------------------------

Use normal dictionary operations to inspect names, add matrices or remove them:

.. code-block:: python

    print(list(demand))
    print(len(demand))
    print(demand.shape)
    print(demand.index)

    if "car" in demand:
        print(demand["car"].sum())

    demand["total"] = demand["car"] + demand["bus"]
    for name, values in demand.items():
        print(name, values.sum())
        
    del demand["total"]

``len(demand)`` returns the number of matrices. ``demand.shape`` is the shape
of each matrix. On a store, reading values through ``items()`` or ``values()``
loads each matrix as it is read. Iterating over names does not load their
values.

Read and change values
----------------------

Put the matrix name first, followed by row and column selections:

.. code-block:: python

    values = demand["car"]
    cell = demand["car", 0, 1]
    column = demand["car", :, 1]
    selected_rows = demand["car", [0, 2], :]

    demand["car", 0, 1] = 42
    demand["car", :, 2] = [1, 2, 3]
    demand["car", :2, :2] = [[0, 5], [6, 0]]
    demand["bus", :, :] *= 1.05

Selections use zero-based positions, not zone IDs. In this example, position
``0`` represents zone ``101``.

When working with an ``AequilibraEMatrix``, selections follow NumPy
indexing. When working with a ``MatrixStore``, selections follow the indexing
supported by the OMX (and h5py) library, which may have different restrictions
on fancy selections. This allows for partial reads and writes to and from
matrices both in memory and on disk.

Whole-matrix reads and basic slices of an ``AequilibraEMatrix`` may share its
memory, fancy NumPy selections may return copies. If an independent array is
needed, use ``demand["car"].copy()``.

Reads from a ``MatrixStore`` always returns a copy. To write changes back,
assign them through the store fancy indexing rather than modifying an array
returned by a read.

Work with the zone index
------------------------

The index must be a one-dimensional sequence of unique, non-negative integer
zone IDs. The same index applies to origins and destinations. The default
index name is ``main_index``. Pass ``index_name`` when a file uses another
mapping name to read it instead.

``index`` is read-only when retrieved. Replace it as a whole to relabel zones:

.. code-block:: python

    demand.index = [1001, 1002, 1003]

The replacement must have the same length. Relabelling does not reorder matrix
values. To change the number of zones, create a new object with the required
index and matrices.

Read and update files
---------------------

A store opens an existing file read-only by default. Use ``mode="r+"`` to
update an existing file, and a ``with`` block to close it when finished:

.. code-block:: python

    with from_file(
        "demand.omx", index_name="zones", mode="r+", store=True
    ) as store:
        first_row = store["car", 0, :]
        store["car", 0, 1] = 42
        store["bus", :, 2] = [1, 2, 3]
        store.metadata["description"] = "Updated morning demand"

These assignments update the file directly because they use fancy index. No
separate ``save_as_omx()`` call is needed.  ``store["car"][:, 2] = ...`` would
only change the returned array, not the file.  Use ``store["car", :, 2] = ...``
instead.

Use ``from_dict(..., store=True)`` with a path and index to create a store from
arrays. ``mode="x"`` prevents replacement of an existing file:

.. code-block:: python

    with from_dict(
        {"car": car, "bus": bus}, index=index, index_name="zones",
        path="new_demand.omx", mode="x", store=True,
    ) as store:
        print(list(store))

A store must stay open while it is being used. If a ``with`` block is not
suitable, close it explicitly with ``store.close()``. An ``AequilibraEMatrix``
does not need to be closed.

To load a whole OMX file into memory, omit ``store=True`` and
``subset=[...]``. ``from_file()`` returns an ``AequilibraEMatrix`` by default:

.. code-block:: python

    from aequilibrae.matrix import from_file

    loaded = from_file("demand.omx", index_name="zones")

    with from_file("demand.omx", index_name="zones", store=True) as store:
        working = store.subset(["car", "bus"])

Providing ``subset`` and ``store=True`` at the same time is not supported. If
using ``store=True``, use ``.subset()`` to load select matrices into a new
``AequilibraEMatrix`` object.

Save matrices and metadata
--------------------------

Use ``save_as_omx()`` to save a matrix. Save all named matrices, or select
which ones to write using the ``subset`` argument:

.. code-block:: python

    demand.metadata["description"] = "Adjusted morning demand"
    demand.save_as_omx("updated_demand.omx")  # All matrices
    demand.save_as_omx("car_demand.omx", subset=["car"])

The file includes the zone mapping and file-level metadata. Existing matrix
names and mappings are not replaced unless ``overwrite=True`` is supplied:

.. code-block:: python

    demand.save_as_omx("updated_demand.omx", overwrite=True)

Prefer a new output file when the original must be preserved. Saving selected
matrices to an existing file does not remove other matrices from the file.

You can supply per-matrix attributes separately when saving:

.. code-block:: python

    demand.save_as_omx(
        "tagged_demand.omx",
        matrix_metadata={"car": {"mode": "car"}, "bus": {"mode": "bus"}},
    )

The ``matrix_metadata`` is translated directly to OMX (and thus HDF5)
attributes for the matrix groups. See the `h5py attributes documentation
<https://docs.h5py.org/en/stable/high/attr.html>` for more information.
    
A store can use these attributes to select an in-memory working set using OMX
attribute queries:

.. code-block:: python

    with from_file("tagged_demand.omx", index_name="zones", store=True) as store:
        car_demand = store.subset([{"mode": "car"}])
        columns = store[{"mode": "car"}, :, 1]
        print(store.omx["car"].attrs["mode"])

Attribute queries through ``store[...]`` return a list of arrays. ``subset()``
creates a new in-memory ``AequilibraEMatrix`` will all matrices that matched
the query. Use ``store.omx`` for advanced OMX-specific operations, or
``store.omx["/"]`` to access the underlying h5py root dataset.

Create matrices from tabular or sparse data
-------------------------------------------

Use ``from_df()`` for an origin-destination table:

.. code-block:: python

    import pandas as pd
    from aequilibrae.matrix import from_df

    trips = pd.DataFrame(
        {"origin": [101, 101, 309], "destination": [205, 205, 101], "car": [2, 3, 7]}
    )
    demand_from_trips = from_df(
        trips, row="origin", col="destination", subset=["car"],
        index=[101, 205, 309], index_name="zones",
    )

Repeated origin-destination pairs are summed. Missing pairs default to zero.
Use ``fill_value`` to specify another value. If no index is supplied, it is
inferred from the sorted unique origin and destination IDs. It's highly
recommend to provide an explicit index in-order to include zones with no
trips. All origin and destination IDs must be present in that index.

Use ``from_scipy()`` when the input is a collection of SciPy sparse arrays:

.. code-block:: python

    from scipy.sparse import csr_array
    from aequilibrae.matrix import from_scipy

    demand_from_sparse = from_scipy(
        {"car": csr_array(car)}, index=index, index_name="zones"
    )

Both helpers return an ``AequilibraEMatrix``. ``from_scipy()`` produces dense
matrices, so allow enough memory for their full square shape.
