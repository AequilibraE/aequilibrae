Understanding the changes in AequilibraE 2.0
============================================

AequilibraE 2.0 is a substantially different software than the 1.X versions that preceded it.

While the change from 0.X to 1.X was mostly due to the maturing of the software feature set with the addition of
public transport modeling capabilities, the change from 1.X to 2.0 includes a wide range of breaking changes
in the API and changes in software behaviour, in addition to a series of new features and performance improvements.


New Algorithms and features
---------------------------


Performance improvements
------------------------

Network skimming
~~~~~~~~~~~~~~~~

The parallelization of the network skimming procedure was re-implemented in Cython, bringing tangible performance
improvements.  Limited benchmarks show performance improvements across various network sizes:

TODO: RUN THESE BENCHMARKS
+-----------------+------------+------------+-----------+--------------------------+
| R01C01_12345678 | Links      | Nodes      | centroids | AequilibraE 2.0 speed-up |
+-----------------+------------+------------+-----------+--------------------------+
| Australia wide  | 30,000,000 | 30,000,000 | 60,000    |                          |
+-----------------+------------+------------+-----------+--------------------------+
| Arkansas        |            |            |           |                          |
+-----------------+------------+------------+-----------+--------------------------+
| Chicago         |            |            |           |                          |
+-----------------+------------+------------+-----------+--------------------------+
| Coquimbo        |            |            |           |                          |
+-----------------+------------+------------+-----------+--------------------------+
|                 |            |            |           |                          |
+-----------------+------------+------------+-----------+--------------------------+



API Changes
--------------------

XXX

Changes in Behaviour
--------------------

Connectivity Analysis
~~~~~~~~~~~~~~~~~~~~~
The Connectivity Analysis procedure no longer returns a list of disconnected OD pairs, but rather a Numpy array
with the nodes that are not connected to the main portion of the network.
This change comes in the heels of replacing the previous Depth-First Search implementation that returned
a full connectivity matrix with the use of a *Connected Components* algorithm, which runs in a few seconds even
for very large networks and is not limited to centroid connectivity, which is substantially more useful for network
diagnostics and editing.

The connectivity analysis is also a feature of the AequilibraE graph, making it substantially easier to use.

.. doctest::

    >>> project = create_example(project_path)
    >>> project.network.build_graphs()

    # We get the graphs for cars and trucks
    >>> graph = project.network.graphs['c']
    >>> graph.disconnected_nodes()