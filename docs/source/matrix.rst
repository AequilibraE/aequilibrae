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
change the source file. ``subset()`` copies the zone index, file-level metadata
and per-matrix attributes as well as the specified matrices. Metadata values
are copied as well.

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

Metadata supplied to ``from_dict()`` is also copied.

Alternatively, start with no named matrices and add them using ``AequilibraEMatrix`` directly:

.. code-block:: python

    empty = AequilibraEMatrix(index=index, index_name="zones")
    empty["car"] = np.zeros(empty.shape)
    empty.update({"bus": bus})

Matrix assignment accepts array-like values with the object's square shape.
``AequilibraEMatrix`` copies supplied arrays so changing an array supplied
earlier does not change the matrix.

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

Use a single string name to read or replace a whole matrix. For selections, use
``.iloc`` with zero-based positions or ``.loc`` with zone IDs. Put the matrix
name last. Both indexers accept a name, a row and name, or a row, column and
name:

.. code-block:: python

    values = demand["car"]
    values = demand.loc["car"]
    values = demand.iloc["car"]
    cell = demand.iloc[0, 1, "car"]
    column = demand.iloc[:, 1, "car"]
    selected_rows = demand.iloc[[0, 2], "car"]

    demand.iloc[0, 1, "car"] = 42
    demand.iloc[:, 2, "car"] = [1, 2, 3]
    demand.iloc[:2, :2, "car"] = [[0, 5], [6, 0]]
    demand.iloc[:, :, "bus"] *= 1.05

In this example, position ``0`` represents zone ``101``. Zone selections use
these IDs instead of positions:

.. code-block:: python

    cell = demand.loc[101, 205, "car"]
    selected_rows = demand.loc[101:205, "car"]
    pairs = demand.loc[[101, 205], [205, 309], "car"]
    demand.loc[[101, 205], [205, 309], "car"] = [10, 20]

Unlike pandas, two lists select matching pairs in NumPy, not all row-column
combinations. Label slices follow the stored zone order and include the stop
label. Steps apply to positions after translating the endpoints. Unknown zone
IDs, including slice endpoints, raise ``KeyError``. Boolean masks, ``...`` and
``None`` pass through unchanged.

Key-only assignments such as ``demand.loc["car"] = values`` replace or create a
whole matrix and require its full shape. Assignments to selections such as
``demand.loc[:, :, "car"] = 0`` use the backend's broadcasting rules.

When working with an ``AequilibraEMatrix``, selections follow NumPy
indexing. When working with a ``MatrixStore``, selections follow the indexing
supported by the OMX (and h5py) library, which may have different restrictions
on fancy selections. In particular, h5py rejects two index lists, unsorted or
repeated positions, and negative slice steps. These limits also apply after
translating zone IDs. This allows for partial reads and writes to and from
matrices both in memory and on disk. Assignments to a selection keep the
matrix's dtype and use NumPy or h5py conversion rules.

Whole-matrix reads and basic slices of an ``AequilibraEMatrix`` may share its
memory, fancy NumPy selections may return copies. If an independent array is
needed, use ``demand["car"].copy()``.

Reads from a ``MatrixStore`` always returns a copy. To write changes back,
assign them through ``store.loc`` or ``store.iloc`` rather than modifying an
array returned by a read.

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

Opening a store with an existing mapping does not rewrite it unless an index
is supplied. Supplying an index requires a writable store. Replacing its index
keeps the mapping's dtype and attributes.

Read and update files
---------------------

``from_file(..., store=True)`` opens an existing file as read-only by default.
Only ``mode="r"`` and ``mode="r+"`` are allowed as the file must already
exist. Use ``mode="r+"`` to update an existing file, and a ``with`` block to
close it when finished:

.. code-block:: python

    with from_file(
        "demand.omx", index_name="zones", mode="r+", store=True
    ) as store:
        first_row = store.iloc[0, "car"]
        store.iloc[0, 1, "car"] = 42
        store.iloc[:, 2, "bus"] = [1, 2, 3]
        store.metadata["description"] = "Updated morning demand"

These assignments update the file directly because they use ``.iloc``. No
separate ``save_as_omx()`` call is needed.  ``store["car"][:, 2] = ...`` would
only change the returned array, not the file.  Use ``store.iloc[:, 2, "car"] = ...``
instead.

Use ``from_dict(..., store=True)`` with a path and index to create a store from
arrays. The default ``mode="x"`` prevents replacement of an existing file:

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

OMX manages the file attributes ``OMX_VERSION``, ``OMX_CREATED_WITH``,
``SHAPE`` and ``TITLE``. They are ignored when loading, copying or saving file
metadata. A store still exposes them through ``metadata``, which is backed by
h5py. Do not change them directly.

The file includes the zone mapping, file metadata and matrix attributes.  An
existing mapping is left unchanged if its zone IDs and order match the object's
index.

Use ``overwrite=True`` to:

* Replace matrices that already exist.
* Change an existing index mapping.
* Write file metadata keys that already exist.

.. code-block:: python

    demand.save_as_omx("updated_demand.omx", overwrite=True)

Updating a mapping keeps its dtype, attributes and length. The new IDs must fit
its dtype, even when ``overwrite=True``.

Prefer a new output file when the original must be preserved. Saving selected
matrices to an existing file does not remove other matrices from the file.
When overwriting a matrix, source attributes replace matching attributes in the
file. Other attributes on that matrix are kept.

Use ``matrix_metadata(name)`` to read or change attributes for an existing matrix:

.. code-block:: python

    demand.matrix_metadata("car")["mode"] = "car"
    demand.matrix_metadata("bus")["mode"] = "bus"

This returns a dictionary in memory and the h5py attributes for a store.
Read-only stores cannot change attributes. For a store, call the method again
after replacing a matrix to get the new dataset's attributes.

Attributes are saved with their matrices:

.. code-block:: python

    demand.save_as_omx("tagged_demand.omx")

Matrix attributes are translated directly to OMX (and thus HDF5) attributes
for the matrix datasets. See the `h5py attributes
documentation <https://docs.h5py.org/en/stable/high/attr.html>` for more
information.
    
A store can use these attributes to select an in-memory working set using OMX
attribute queries:

.. code-block:: python

    with from_file("tagged_demand.omx", index_name="zones", store=True) as store:
        car_demand = store.subset([{"mode": "car"}])
        column = car_demand.iloc[:, 1, "car"]
        print(store.matrix_metadata("car")["mode"])

Attribute queries are only supported by ``subset()`` and the ``subset``
arguments when loading matrices. They are not supported by ``store[...]``,
``.loc`` or ``.iloc``. ``subset()`` creates a new in-memory
``AequilibraEMatrix`` with all matrices that matched the query. Use
``store.omx`` for advanced OMX-specific operations, or ``store.omx["/"]`` to
access the underlying h5py root dataset.

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
