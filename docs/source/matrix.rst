Working with matrices
=====================

AequilibraE provides four objects for working with named matrices:

* ``AequilibraeMatrix`` holds matrices in memory and is backed by a dictionary.
  Prefer this object for dense calculations and other in-memory work.
* ``AequilibraeSparseMatrix`` holds SciPy sparse arrays in memory and keeps
  their sparse formats and dtypes.
* ``MatrixStore`` is backed by an OMX file. Use it to read and update dense
  matrices on disk through a similar interface to the in-memory object.
* ``SparseMatrixStore`` reads and writes whole sparse matrices using a CSR
  extension to OMX.

All four objects associate names with square matrices. All matrices in an object
share the same zone index, which gives the zone IDs in row and column order.
Use the ``from_file()``, ``from_dict()``, ``from_df()`` and ``from_scipy()``
functions to create the appropriate object from your source data, or the
``AequilibraeMatrix`` constructor directly.

These objects replace the previous ``AequilibraeMatrix`` which mixed both
roles. For a migration guide, see :doc:`matrix_api_migration`.

``AequilibraeMatrix``, ``AequilibraeSparseMatrix``, ``MatrixStore`` and
``SparseMatrixStore`` implement the ``MutableMapping`` protocol and behave
very similarly to a dictionary. You can use mapping operations including
``.keys()``, ``.values()``, ``.items()`` and ``.update()``.

Create an in-memory working set
-------------------------------

The recommended workflow is to open an OMX file with ``from_file(...,
subset=[...])``, which returns an ``AequilibraeMatrix``. If you don't know
which matrices you need ahead of time, provide ``store=True`` to ``from_file``
instead of ``subset`` to obtain a ``MatrixStore``. Inspect the file, then use
its ``subset()`` method to select the matrices needed for a calculation. It
returns a new, independent ``AequilibraeMatrix``. Work on that object in memory,
then save the result when needed.

The following example assumes that ``demand.omx`` contains matrices named
``car`` and ``bus`` and a zone mapping named ``zones``. ``index_name`` is
optional when the file has exactly one zone mapping. If it has multiple
mappings, ``from_file()`` selects ``main_index`` when present, otherwise, pass
``index_name`` explicitly.

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
are copied as well. Missing names raise ``KeyError``. When selecting from a
file, the error includes a hint if the name exists in the other storage group.
This applies to store subsets, loading subsets and saving subsets from a store.

Use ``MatrixStore`` for manipulating files rather than working with it as if it
was in-memory. It reads the matrix from disk each time it is accessed. For this
reason, arrays obtained from it are copies. Mutating them changes the copy
rather than the file itself. Assign through ``.mat`` or ``.imat`` to change
the file (see ``Read and change values`` below).

Create matrices in memory
-------------------------

Use ``from_dict()`` when arrays are already available:

.. code-block:: python

    import numpy as np
    from aequilibrae.matrix import AequilibraeMatrix, from_dict

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

Alternatively, start with no named matrices and add them using ``AequilibraeMatrix`` directly:

.. code-block:: python

    empty = AequilibraeMatrix(index=index, index_name="zones")
    empty["car"] = np.zeros(empty.shape)
    empty.update({"bus": bus})

Matrix assignment accepts array-like values with the object's square shape.
Matrix values must use integer, floating-point, boolean or complex dtypes.
This limit does not apply to metadata.  Whole-matrix assignment uses the
supplied dtype, even when replacing an existing matrix. ``AequilibraeMatrix``
copies supplied arrays so changing an array supplied earlier does not change
the matrix.

Inspect and manage named matrices
---------------------------------

Matrix names must be non-empty strings, not paths. Names cannot contain a slash,
backslash or null character, and cannot be ``.`` or ``..``.

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

``clear()`` removes all named matrices and their attributes without reading
matrix values. It keeps the zone index and file-level metadata.

``setdefault(name, values)`` adds the supplied matrix only if the name is
missing, then returns the matrix that would be read for that key. Unlike
``dict.setdefault``, the default of ``values=None`` is not a valid value.

Equality (``==``) compares matrix names and values, zone IDs, the index name,
file-level metadata and per-matrix attributes. NaNs are considered
equal. Matrix insertion order and numeric dtypes do not affect equality. The
OMX-managed file attributes listed below are ignored. For a ``MatrixStore``,
the values must be read from disk.

Read and change values
----------------------

Use a single string name to read or replace a whole matrix. For selections, use
``.imat`` with zero-based positions or ``.mat`` with zone IDs. Both indexers
accept a name, a name and row, or a name, row and column:

.. code-block:: python

    # all the same
    values = demand["car"]
    values = demand.mat["car"]
    values = demand.imat["car"]
    
    cell = demand.imat["car", 0, 1]
    column = demand.imat["car", :, 1]
    selected_rows = demand.imat["car", [0, 2]]

    demand.imat["car", 0, 1] = 42
    demand.imat["car", :, 2] = [1, 2, 3]
    demand.imat["car", :2, :2] = [[0, 5], [6, 0]]
    demand.imat["bus", :, :] *= 1.05

In this example, position ``0`` represents zone ``101``. Zone selections use
these IDs instead of positions:

.. code-block:: python

    cell = demand.mat["car", 101, 205]
    selected_rows = demand.mat["car", 101:205]
    pairs = demand.mat["car", [101, 205], [205, 309]]
    demand.mat["car", [101, 205], [205, 309]] = [10, 20]

Label slices follow the stored zone order and include the stop label to allow
simple assess to a range of zones. Steps apply to positions after translating
the endpoints, not the zone IDs. Unknown zone IDs, including slice endpoints,
raise a ``KeyError``.

Key-only assignments such as ``demand.mat["car"] = values`` replace or create a
whole matrix and require its full shape. Assignments to selections such as
``demand.mat["car", :, :] = 0`` use the backend's broadcasting rules.

When working with an ``AequilibraeMatrix``, selections follow NumPy
indexing. When working with a ``MatrixStore``, selections follow the indexing
supported by the OMX (and h5py) library, which may have different restrictions
on fancy selections. Selections are passed to the underlying object after
translating zone IDs, so support for a selection depends on that object.  This
allows for partial reads and writes to and from matrices both in memory and on
disk. Assignments to a selection keep the matrix's dtype and use NumPy or h5py
conversion rules. ``SparseMatrixStore`` only supports whole-matrix reads and
writes. Selections through ``mat`` or ``imat`` raise ``NotImplementedError``.

Whole-matrix reads and basic slices of an ``AequilibraeMatrix`` may share its
memory, fancy NumPy selections may return copies. If an independent array is
needed, use ``demand["car"].copy()``.

Reads from a ``MatrixStore`` always returns a copy. To write changes back,
assign them through ``store.mat`` or ``store.imat`` rather than modifying an
array returned by a read.

Work with the zone index
------------------------

The index must be a one-dimensional sequence of unique, non-negative integer
zone IDs from ``0`` to ``2**64 - 1``. IDs are stored in memory as ``uint64``.
Booleans and floating-point values are not accepted, even floats such as
``1.0``. The same index applies to origins and destinations. The default index
name is ``main_index``. Pass ``index_name`` when a file uses another mapping
name to read it instead. Index names follow the same rules as matrix names:
they must be non-empty strings, not paths.

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
        first_row = store.imat["car", 0]
        store.imat["car", 0, 1] = 42
        store.imat["bus", :, 2] = [1, 2, 3]
        store.metadata["description"] = "Updated morning demand"

These assignments update the file directly because they use ``.imat``. No
separate ``save_as_omx()`` call is needed.  ``store["car"][:, 2] = ...`` would
only change the returned array, not the file.  Use ``store.imat["car", :, 2] =
...`` instead.

Use ``from_dict(..., store=True)`` with a path and index to create a store from
arrays. The default ``mode="x"`` prevents replacement of an existing file:

.. code-block:: python

    with from_dict(
        {"car": car, "bus": bus}, index=index, index_name="zones",
        path="new_demand.omx", mode="x", store=True,
    ) as store:
        print(list(store))

A store must stay open while it is being used. If a ``with`` block is not
suitable, close it explicitly with ``store.close()``. An ``AequilibraeMatrix``
does not need to be closed.

To load a whole OMX file into memory, omit ``store=True`` and
``subset=[...]``. ``from_file()`` returns an ``AequilibraeMatrix`` by default:

.. code-block:: python

    from aequilibrae.matrix import from_file

    loaded = from_file("demand.omx", index_name="zones")

    with from_file("demand.omx", index_name="zones", store=True) as store:
        working = store.subset(["car", "bus"])

Providing ``subset`` and ``store=True`` at the same time is not supported. If
using ``store=True``, use ``.subset()`` to load select matrices into a new
``AequilibraeMatrix`` object.

Save matrices and metadata
--------------------------

Use ``save_as_omx()`` to save a matrix. The default ``mode="x"`` creates a new
file and raises ``FileExistsError`` if the file already exists. Save all named
matrices, or use ``subset`` to select them by name or by their attributes:

.. code-block:: python

    demand.metadata["description"] = "Adjusted morning demand"
    demand.save_as_omx("updated_demand.omx")  # All matrices
    demand.save_as_omx("car_demand.omx", subset=["car"])

Attribute queries use the source object's matrix attributes. You can mix names
and queries, and each selected matrix is saved once:

.. code-block:: python

    demand.matrix_metadata("car")["mode"] = "car"
    demand.save_as_omx("car_by_mode.omx", subset=[{"mode": "car"}])
    demand.save_as_omx("selected_demand.omx", subset=["bus", {"mode": "car"}])

All attributes in a query must match. If no matrices match, the zone mapping
and file metadata are still written.

OMX manages the file attributes ``OMX_VERSION``, ``OMX_CREATED_WITH``,
``SHAPE`` and ``TITLE``. They are ignored when loading, copying, saving or
comparing file metadata. A store still exposes them through ``metadata``, which
is backed by h5py. Do not change them directly.

The file includes the zone mapping, file metadata and matrix attributes.  Use
``mode`` to choose how to write the file:

* ``"x"`` or ``"w-"``: create a new file, failing if it already exists.
* ``"r+"``: update an existing file, failing if it does not exist.
* ``"a"``: update an existing file, or create it if it does not exist.
* ``"w"``: replace the whole file, or create it if it does not exist.

.. code-block:: python

    demand.save_as_omx("updated_demand.omx", mode="r+")

The zone mapping and supplied metadata are always written, be careful when
updating existing files. Saving selected matrices with ``mode="r+"`` or
``mode="a"`` does not remove other matrices from the file. In contrast,
``mode="w"`` removes all previous contents, even when saving a subset. When
overwriting a matrix, new attributes will replace matching attributes in the
file. Other attributes on that matrix are not modified.

Use ``matrix_metadata(name)`` to read or change attributes for an existing matrix:

.. code-block:: python

    demand.matrix_metadata("car")["mode"] = "car"
    demand.matrix_metadata("bus")["mode"] = "bus"

This returns a dictionary in memory and the h5py attributes for a store.
Read-only stores cannot change attributes. For a store, call the method again
after changing a matrix's dtype to get the new dataset's attributes.

For both ``metadata`` and ``matrix_metadata(name)``, replace each attribute
value as a whole. Nested edits, such as ``demand.metadata["periods"][0] = 9``,
are not supported.

.. code-block:: python

    demand.metadata["periods"] = [1, 2, 3]
    demand.matrix_metadata("car")["periods"] = [1, 2]

Attributes are saved with their matrices:

.. code-block:: python

    demand.save_as_omx("tagged_demand.omx")

Matrix attributes are translated directly to OMX (and thus HDF5) attributes
for the matrix datasets. See the `h5py attributes
documentation <https://docs.h5py.org/en/stable/high/attr.html>` for more
information.
    
Both objects can use these attributes to select an in-memory working set using
attribute queries:

.. code-block:: python

    with from_file("tagged_demand.omx", index_name="zones", store=True) as store:
        car_demand = store.subset([{"mode": "car"}])
        column = car_demand.imat["car", :, 1]
        print(store.matrix_metadata("car")["mode"])

    car_demand = demand.subset([{"mode": "car"}])

All attributes within a query must match, and multiple queries select the
combined matches, with each matrix included once.

Attribute queries are supported by ``subset()`` and the ``subset`` arguments
when loading or saving matrices. They are not supported by ``store[...]``,
``.mat`` or ``.imat``. ``subset()`` creates a new in-memory
``AequilibraeMatrix`` with all matrices that matched the query. Use
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
trips. All origin and destination IDs must be present in that index. These IDs
must follow the zone-ID rules above, whether the index is supplied or inferred.

Use ``from_scipy()`` when the input is a collection of SciPy sparse arrays:

.. code-block:: python

    from scipy.sparse import csr_array
    from aequilibrae.matrix import from_scipy

    demand_from_sparse = from_scipy(
        {"car": csr_array(car)}, index=index, index_name="zones"
    )

``from_df()`` returns an ``AequilibraeMatrix``. By default, ``from_scipy()`` returns
an ``AequilibraeSparseMatrix`` and copies the sparse arrays without changing their
formats or dtypes. You can also create an empty sparse collection with
``AequilibraeSparseMatrix(index=index)`` and assign named sparse arrays.
Whole-matrix reads return the stored sparse array. Selections through ``mat``
and ``imat`` follow that sparse format's SciPy indexing rules.

``save_as_omx()`` converts sparse arrays to CSR for storage without changing
them in memory. Each matrix uses a group at ``/sparse/<name>`` with datasets
``row_index`` (CSR row pointers), ``col_index`` (column positions) and
``data``. Other group attributes hold matrix metadata. Zone IDs remain under
``/lookup``. This is an extension to OMX, so standard OMX readers will not see
these sparse matrices.

Load these files with ``AequilibraeSparseMatrix.from_omx()`` or
``from_file(..., sparse=True)``. These only read ``/sparse`` and ignore dense
matrices in ``/data``. Dense and sparse matrices may not have the same name.

Work with sparse matrices on disk
--------------------------------

Use ``SparseMatrixStore`` or ``from_file(..., sparse=True, store=True)`` to
work with sparse matrices without loading the whole collection. Each matrix
read returns an independent CSR array. Assign the whole array back to save
changes. Partial reads and writes are not supported.

.. code-block:: python

    from aequilibrae.matrix import from_file, from_scipy

    with from_scipy(
        {"car": csr_array(car)}, index=index, index_name="zones",
        path="sparse_demand.omx", store=True,
    ) as store:
        store.matrix_metadata("car")["mode"] = "car"

    with from_file("sparse_demand.omx", sparse=True, store=True, mode="r+") as store:
        values = store["car"]
        values *= 2
        store["car"] = values
        working = store.subset([{"mode": "car"}])

``from_scipy(..., store=True)`` writes CSR arrays and keeps their dtypes. Its
file mode defaults to ``"x"``, which fails if the file already exists.
``from_file(..., store=True)`` opens read-only unless ``mode="r+"`` is supplied.

Sparse stores share the file metadata and zone mappings with dense stores. A
store's ``subset()`` returns an independent ``AequilibraeSparseMatrix`` which
can be used after the store closes.

Native sparse buffers
---------------------

Use ``SparseMatrix`` to pack a sparse collection for C++ code:

.. code-block:: python

    from aequilibrae.matrix.cython.sparse_matrix import SparseMatrix

    packed = SparseMatrix.from_matrix(demand_from_sparse)
    shared = packed.to_matrix(demand_from_sparse.index, index_name="zones")

``from_matrix()`` copies the union of all stored positions into one CSR index.
All fields use that index, with zero for positions missing from a source field.
Stored zeros are kept and duplicate positions are summed. The source arrays,
their formats and their values are not changed. Zone IDs and metadata are not
retained in the packed buffer.

All packed values use one dtype, either float32 or float64. Source dtypes must
match unless you pass ``dtype=`` to convert them. Complex values are rejected.
Empty buffers are allowed.

``to_matrix()`` returns an ``AequilibraeSparseMatrix`` with CSR arrays built
from the row pointers, column indices and shared value buffer. Named data
arrays may be strided. Value changes at existing positions are shared with the
packed buffer. The arrays keep the buffer alive. Adding or removing stored
positions is not supported by this shared view. The supplied zone index is
copied and normal whole-matrix assignment still copies values.

Cython code can borrow a ``CppSparseMatrix[float]`` or
``CppSparseMatrix[double]`` with ``packed.view[float]()`` or
``packed.view[double]()``. The type must match the packed dtype. The view can
be used without the GIL, but its owner must remain alive while it is in use.
