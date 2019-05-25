API Documentation
==================
An attentive reader will notice that the documentation of different modules are quite different, and that is due to the
fact that API documentation came as an afterthought during AequilibraE's development. On that note, documentation is
under active development and is probably the best place for a new user to start contributing to it.

.. note::
   If the documentation here is not sufficient, you can always resort to the mailing list: aequilibrae@googlegroups.com


Paths module
-------------

.. automodule:: aequilibrae.paths
   :members:

The Aequilibrae Graph
---------------------

The AequilibraE Graph is the object that holds all the information needed for traffic assignment. It holds not only
the links and nodes from the network, but also the other fields needed for skimming, volume delay functions, etc.

Object Variables
~~~~~~~~~~~~~~~~

``num_links`` (**int**): Number of directed links in the graph

``num_nodes`` (**int**): Number of nodes in the graph

``network`` (**np.recarray**): Numpy record array with arbitrary number of fields (and titles of columns)
corresponding to each link (*dimension*: num_links)

``nodes_fs`` (**np.ndarray**): Numpy array with indices of each node in the forward star (*dimension*:
num_nodes + 1)

``cost`` (**str**): Name of the field to be minimized.

``skims`` (**list**): Name of the skim fields

``description`` (**str**): Description of the graph (*optional*)


.. automodule:: aequilibrae.paths.graph
   :members:


Matrix module
-------------

The AequilibraeMatrix is a highly efficient matrix format that underlines all AequilibraE computation

It is capable of storing up to 256 different matrices (cores) per file, can have multiple indices to support matrix
aggregation and metadata of up to 144 caracters

It is based on NumPy's memory-mapped arrays, so it is highly efficient, and the format as a memory-blog on disk
makes it possible for other software to read the matrices as well.

.. automodule:: aequilibrae.matrix.aequilibrae_matrix
   :members:
