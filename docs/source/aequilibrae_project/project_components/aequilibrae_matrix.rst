.. _all_about_aeq_matrices:

AequilibraE Matrix
==================

AequilibraE provides ``AequilibraeMatrix`` for in-memory matrix work and
``MatrixStore`` for reading and updating OMX files on disk. For the complete
API, see :doc:`../../matrix`.

``AequilibraeMatrix``
---------------------

Create an in-memory matrix by providing its zone index, then assign named
matrix arrays directly:

.. doctest::

    >>> import numpy as np
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from aequilibrae.matrix import AequilibraeMatrix, from_file

    >>> index = np.arange(1, 6, dtype=np.uint64)
    >>> values = np.ones((5, 5), dtype=np.float32)
    >>> mat = AequilibraeMatrix(index=index, index_name="zones")
    >>> mat["only_ones"] = values

    # Inspect its names and zone index.
    >>> list(mat)
    ['only_ones']
    >>> mat.index
    array([1, 2, 3, 4, 5], dtype=uint64)
    >>> mat["only_ones"]
    array([[1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.]], dtype=float32)

Use the named array directly in computations. ``subset()`` creates an
independent in-memory copy containing only the requested matrices:

.. doctest::

    >>> working = mat.subset(["only_ones"])
    >>> working["only_ones"].shape
    (5, 5)

Save an in-memory matrix as OMX with ``save_as_omx()`` and load it with
``from_file()``. A single OMX index is inferred automatically; pass
``index_name`` when the file has multiple indices and no ``main_index``.

.. doctest::

    >>> matrix_tmpdir = TemporaryDirectory()
    >>> omx_path = Path(matrix_tmpdir.name) / "matrix_example.omx"
    >>> mat.save_as_omx(omx_path)
    >>> loaded = from_file(omx_path, subset=["only_ones"])
    >>> loaded["only_ones"].shape
    (5, 5)

A ``MatrixStore`` keeps data on disk and is useful when loading every matrix
into memory is undesirable. Select a working subset directly with
``from_file(..., subset=...)`` or use a store to inspect the file before
selecting matrices:

.. doctest::

    >>> with from_file(omx_path, store=True) as store:
    ...     list(store)
    ['only_ones']

OpenMatrix (OMX)
----------------

AequilibraE uses OMX files as a standard format for storing its matrices. If
you're wondering what OMX is and what it stands for, this section is for you.
The text in this section is borrowed from
`OpenMatrix Wiki page <https://github.com/osPlanning/omx/wiki>`_.

The OpenMatrix file format (or simply OMX) is a standard matrix format for storing and
transferring matrix data across different models and software packages, intended to make
the model development easier. It is a file capable of storing more than one matrices
at a time, including multiple indexes/lookups, and attributes (key/value pairs) for matrices and
indexes.

There are APIs in different programming languages that allow you to use OMX.
In Python, we use the ``omx-python`` library. See its project page for a
`brief tutorial <https://github.com/osPlanning/omx-python?tab=readme-ov-file#quick-start-sample-code>`_.
