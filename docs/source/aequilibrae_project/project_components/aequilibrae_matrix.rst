.. _all_about_aeq_matrices:

AequilibraE Matrix
==================

The AequilibraEMatrix class is the AequilibraE vehicle to all things matrices, and in the 
following sections we'll cover the main points regarding them.

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
    
    >>> num_zones = 5
    >>> index = np.arange(1, 6, dtype=np.int32)
    >>> mtx = np.ones((5, 5), dtype=np.float32)
    >>> names = ["only_ones"]

    >>> mat = AequilibraeMatrix()
    >>> mat.create_empty(zones=num_zones, matrix_names=names) #memory_only parameter defaults to True

    # `memory_only` parameter can be changed to `False` case you want to save the matrix in disk.
	# This would, however, result in a file format that is being deprecated and will not be available on AequilibraE 2.0

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

More than dealing with stored project data, AequilibraE matrices are objects necessary to run procedures,
such as traffic assignment. Since a matrix object can hold multiple matrices (i.,e. _matrix cores_),
it is necessary to specify which matrices will be used in computation, dubbed a computational view in 
AequilibraE, which sets matrix data in memory in a way it can be used in parallelized algorithms.

Case you're using matricial data from an OMX file, this step also loads the data to memory.

.. code-block:: python

    >>> mat.computational_view(["only_ones"])

You can also export AequilibraE matrices, with your chosen set of _matrix cores_, to different file 
formats, such as CSV and OMX.

.. code-block:: python

    >>> mat.export(Path(my_folder_path) / 'my_new_omx_file.omx')

    >>> mat.export(Path(my_folder_path) / 'my_new_csv_file.csv')


To avoid inconsistencies, once open, the same AequilibraE matrix can only be used once at a time in different
procedures. To do so, you have to close the matrix.

.. code-block:: python

    >>> mat.close()

AequilibraE matrices in disk can be reused and loaded once again.

.. code-block:: python

    >>> mat = AequilibraeMatrix()
    >>> mat.load(Path(my_folder_path) / 'my_new_omx_file.omx')

    >>> mat.get_matrix("only_ones") # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([[1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.]])

.. seealso::

    :func:`aequilibrae.matrix.AequilibraeMatrix`
        Class documentatiom
    :ref:`plot_assignment_without_model`
        Usage example 

OpenMatrix (OMX)
----------------

AequilibraE uses OMX files as a standard format for storing its matrices. 
If you're wondering what is OMX and what does
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
to OMX, and better understand how it works.
