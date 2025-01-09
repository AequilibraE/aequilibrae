.. _all_about_aeq_matrices:

AequilibraE Matrix
==================

AequilibraE matrices are very useful objects that allow you to make the most with AequilibraE.
In the following sections, we'll cover the main points regarding them.

``AequilibraeMatrix``
---------------------

This class allows the creation of a memory instance for a matrix, that can be used to load an existing
matrix to the project, or to create a new one. 

There are three ways of creating an ``AequilibraeMatrix``:

* from an OMX file;
* from a trip list, which is nothing more than a CSV file containing the origins, destinations, and trip cores; 
* from an empty matrix. In this case, the data type must be one of the following NumPy data types: 
  ``np.int32``, ``np.int64``, ``np.float32``, ``np.float64``.
    
.. code-block:: python

    >>> from aequilibrae.matrix import AequilibraeMatrix
    
    >>> file = os.path.join(my_folder_path, "path_to_my_matrix.aem")
    >>> num_zones = 5
    >>> index = np.arange(1, 6, dtype=np.int32)
    >>> mtx = np.ones((5, 5), dtype=np.float32)
    >>> names = ["only_ones"]

    >>> mat = AequilibraeMatrix()
    >>> mat.create_empty(file_name=file, zones=num_zones, matrix_names=names)

    # `memory_only` parameter can be changed to `True` case you want to save the matrix in disk.

    # Adds the matrix indexes, which are going to be used for computation
    >>> mat.index[:] = index[:]

    # Adds the matricial data stored in `mtx` to a matrix named "only_ones"
    >>> mat.matrix["only_ones"][:,:] = mtx[:,:]

The following methods allow you to check the data in you AequilibraE matrix.

.. code-block:: python

    >>> mat.cores # displays the number of cores in the matrix
    1

    >>> mat.names # displays the names of the matrices
    ['only_ones']
    
    >>> mat.index # displays the IDs of the indexes
    array([1, 2, 3, 4, 5])
    
    # To return an array with the selected matrix data
    >>> mat.get_matrix("only_ones") # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([[1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.]])

More than storing project data, AequilibraE matrices are objects necessary to run procedures,
such as traffic assignment. To do so, one must create a computational view of the matrix, which
allows AequilibraE matrices to be used in parallelized algorithms. It is possible to create a 
computational view for more than one matrix at a time.

Case you're using matricial data from an OMX file, this step is mandatory to load the data to memory,
otherwise the matrix is useless in other procedures.

.. code-block:: python

    >>> mat.computational_view(["only_ones"])

You can also export AequilibraE matrices to another file formats, such as CSV and OMX. When exporting
to a OMX file, you can choose the cores os the matrix you want to save, although this is not the case
for CSV file, in which all cores will be exported as separate columns in the output file.

.. code-block:: python

    >>> mat.export(os.path.join(my_folder_path, 'my_new_omx_file.omx'))

    >>> mat.export(os.path.join(my_folder_path, 'my_new_csv_file.csv'))

The ``export`` method also allows you to change your mind and save your AequilibraE matrix into an AEM
file, if it's only in memory.

.. code-block:: python

    >>> mat.export(os.path.join(my_folder_path, 'my_new_aem_file.aem'))

.. is there a better name rather than error?

To avoid errors, once open, the same AequilibraE matrix can only be used once at a time in different
procedures. To do so, you have to close the matrix, to remove it from memory and flush the data to disk,
or to close the OMX file, if that's the case.

.. code-block:: python

    >>> mat.close()

AequilibraE matrices in disk can be reused and loaded once again.

.. code-block:: python

    >>> mat = AequilibraeMatrix()
    >>> mat.load(os.path.join(my_folder_path, 'my_new_aem_file.aem'))

    >>> mat.get_matrix("only_ones") # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    memmap([[1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]])

.. important::

    File extension for AequilibraE matrices is **AEM**.

.. seealso::

    :func:`aequilibrae.matrix.AequilibraeMatrix`
        Class documentatiom
    :ref:`plot_assignment_without_model`
        Usage example 

OpenMatrix (OMX)
----------------

AequilibraE can handle OMX files, but if you're wondering what is OMX and what does
it stand for, this section is for you. The text in this section is borrowed from 
`OpenMatrix Wiki page <https://github.com/osPlanning/omx/wiki>`_.

The OpenMatrix file format (or simply OMX) is a standard matrix format for storing and
transferring matrix data across different models and software packages, intended to make
the model development easier. It is a file capable of storing more than one matrices
at a time, including multiple indexes/lookups, and attributes (key/value pairs) for matrices and
indexes.

There are APIs in different programming languages that allow you to use OMX. In Python, we use
``omx-python`` library. In its project page, you can find a 
`brief tutorial <https://github.com/osPlanning/omx-python?tab=readme-ov-file#quick-start-sample-code>`_
to OMX, and better understand how does it work.

Creating an AequilibraE matrix from an OMX file is pretty straightforward.

.. code-block:: python

    >>> file_path = os.path.join(my_folder_path, "path_to_new_matrix.aem")
    >>> omx_path = os.path.join(my_folder_path, "my_new_omx_file.omx")

    >>> omx_mat = AequilibraeMatrix()
    >>> omx_mat.create_from_omx(omx_path, file_path)

    >>> mat.get_matrix("only_ones") # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    memmap([[1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]])
