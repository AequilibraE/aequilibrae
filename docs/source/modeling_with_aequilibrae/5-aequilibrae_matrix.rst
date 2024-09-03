AequilibraE Matrix
==================

``aequilibrae.matrix`` presents different modules that allow you to make the most
with AequilibraE, and in the following sections, we'll cover the main points regarding them.

``AequilibraeMatrix``
---------------------

This class allows the creation of a memory instance for a matrix, that can be used to load an existing
matrix to the project, or to create a new one. This matrix can be saved into an external file later.

There are three ways of creating an ``AequilibraeMatrix``:

* from an OMX file;
* from a trip list, which is nothing more than a CSV file containing the origins, destinations, and trip cores; 
* from an empty matrix. In this case, the data type must be one of the following NumPy data types: 
  ``np.int32``, ``np.int64``, ``np.float32``, ``np.float64``.

.. code-block:: python

    from aequilibrae.matrix import AequilibraeMatrix

    file = "folder/path_to_my_matrix.aem"
    num_zones = 10

    mat = AequilibraeMatrix()
    mat.create_empty(file_name=file, zones=num_zones, memory_only=False)

    # `memory_only` parameter can be changed to `True` case you want to save the matrix in disk.

    # Adds the matrix index, which are XXXXXXXXXXXX
    mat.index[:] = index[:]
    # Adds the matricial data stored in `mtx` to a matrix named "my_matrix"
    mat.matrix["my_matrix"][:,:] = mtx[:,:]

More than storing project data, AequilibraE matrices are objects necessary to run procedures,
such as traffic assignment. To do so, one must create a ``computational_view`` of the matrix, which
allows AequilibraE matrices to be used in parallelized algorithms. It is possible to create a 
``computational_view`` for more than one matrix at a time

.. code-block:: python

    mat.computational_view(["my_matrix"])

.. important::

    File extension for AequilibraE matrices is **AEM**.

.. seealso::

    :func:`aequilibrae.matrix.AequilibraeMatrix`
        Documentation for ``AequilibraeMatrix`` class

    :ref:`plot_assignment_without_model`
        Usage example 

``AequilibraeData``
-------------------

The ``AequilibraeData`` class allows the creation of datasets, which are objects necessary to
run procedures such as traffic distritubion. 

.. code-block:: python

    import numpy as np
    from aequilibrae.matrix import AequilibraeData

    num_entries = 10

    args = {"file_path": "folder/path_to_my_dataset.aed",
            "entries": num_entries,
            "field_names": ["origins", "destinations"],
            "data_types": [np.float64, np.float64],
            "memory_mode": True}

    # `memory_mode` parameter can be changed to `False` case you don't want to save the matrix in disk.

    dataset = AequilibraeData()
    dataset.create_empty(**args)

    # Adds the dataset indexes
    dataset.index[:] = mtx.index[:]
    # Adds the dataset field's data
    dataset.origins[:] = example_vector_1[:]
    dataset.destinations[:] = example_vector_2[:]

An AequilibraE dataset can be exported to CSV or SQLite formats. Please notice that when setting the ``file_name``,
it must contain one of the following extensions: ``csv``, ``sqlite3``, ``sqlite`` or ``db``, and if using a SQLite
table, the ``table_name`` field is mandatory.

.. code-block:: python

    dataset.export(file_name="folder/path_to_my_dataset_database.db", 
                   table_name="table_name_to_be_saved")

It is possible to reuse AequilibraE datasets that were stored in disk in an AED file.

.. code-block:: python

    dataset.load(file_path="folder/path_to_my_dataset.aed")

.. important::

    File extension for AequilibraE datasets is **AED**.

.. seealso::

    :func:`aequilibrae.matrix.AequilibraeData`
        Documentation for ``AequilibraeData`` class

    :ref:`example_usage_distribution`
        Usage example
