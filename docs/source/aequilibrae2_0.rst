Understanding the changes in AequilibraE 2.0
============================================

AequilibraE 2.0 is a substantially different software than the 1.X versions that preceded it.

While the change from 0.X to 1.X was mostly due to the maturing of the software feature set with the addition of
public transport modeling capabilities, the change from 1.X to 2.0 includes a wide range of breaking changes
in the API and changes in software behaviour, in addition to a series of new features and performance improvements.


New Algorithms and features
---------------------------

Gravity model calibration
~~~~~~~~~~~~~~~~~~~~~~~~~

``GravityCalibration`` now calibrates all three deterrence functions, including ``GAMMA``, by maximum likelihood.
The calibrated model reproduces the observed average trip cost (``EXPO``), the observed average logarithm of the trip
cost (``POWER``), or both (``GAMMA``). See :doc:`distribution_procedures/classes` for details.

Performance improvements
------------------------

Network skimming
~~~~~~~~~~~~~~~~

The parallelization of the network skimming procedure was re-implemented in Cython, bringing tangible performance
improvements for small and medium networks, which cover a significant amount of the real-world networks
in use by transportation planning agencies in 2026.

TODO: RUN THESE BENCHMARKS
+-----------------+------------+-------------+------------+-----------+--------------------------+
| R01C01_12345678 | Links      | Graph Links | Nodes      | centroids | AequilibraE 2.0 speed-up |
+-----------------+------------+-------------+------------+-----------+--------------------------+
| Arkansas        |   300,146  |  591,489    |  243,059   |   6,445   |          0%              |
+-----------------+------------+-------------+------------+-----------+--------------------------+
| Australia wide  | 4,113,479  |  7,760,714  | 30,000,000 | 60,000    |          0%              |
+-----------------+------------+-------------+------------+-----------+--------------------------+
| Chicago         |   39,018   |   39,018    |  12,979    |  1,790    |         12%              |
+-----------------+------------+-------------+------------+-----------+--------------------------+
| Coquimbo        |   19,983   |   34,546    |  15,724    |  133      |         31%              |
+-----------------+------------+-------------+------------+-----------+--------------------------+
| Lyon            |  253,305   |  468,761    |  179,940   |  4,617    |          0%              |
+-----------------+------------+-------------+------------+-----------+--------------------------+

The following conditions apply to this benchmark:

- The skimming procedure was run for a single mode and a single cost variable (i.e. "distance" for cars).
- It does not include the time taken to build the graph or to save the result to disk.
- The benchmarking was run on a desktop PC equipped with a intel i9 285K with 192GB of RAM, and
using 8 threads for the parallelization (this CPU's *performance cores*).
- The speed-up procedure considers the ratio between the median time for 10 runs.

Iterative proportional fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The IPF no longer rescales the whole matrix twice per iteration. It updates one factor per row and one per column
in a single read-only pass over the seed matrix, and applies them once at the end. For matrices of 2,500 zones or
more, this is between 6 and 9 times faster than in AequilibraE 1.X, both on a single thread and on all threads of
a 24-core workstation. See :doc:`distribution_procedures/IPF_benchmark` for details.

The balancing factors are also available through ``ipf_factors``, which does not modify the seed matrix.


API Changes
--------------------

.. toctree::
   :maxdepth: 1

   table_api_migration
   matrix_api_migration

Changes in Behaviour
--------------------

Iterative proportional fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Rows with a zero target total are now emptied. They used to keep their seed values, so the balanced matrix did
  not add up to the targets.
* The number of iterations returned by ``ipf_core`` is the number of iterations performed. It used to be one less.

Gravity models
~~~~~~~~~~~~~~

* The previous calibration's secant update used the latest parameter twice, so it stopped after two iterations,
  reporting convergence, with an approximate result. Results of ``EXPO`` and ``POWER`` calibrations change.
* Calibration of the ``POWER`` function now reproduces the observed average logarithm of the trip cost, its maximum
  likelihood estimate, instead of the observed average trip cost.
* Observed trips in cells the deterrence function cannot reproduce (missing cost, zero cost for ``POWER`` and
  ``GAMMA``, or at or above the maximum trip length) are left out of the calibration.
* ``GravityApplication`` applied the maximum trip length to the deterrence values instead of the cost. It now
  removes trips with a cost at or above the limit.
* ``GravityApplication`` no longer rescales the attractions column of the vectors it receives.

Connectivity Analysis
~~~~~~~~~~~~~~~~~~~~~
The Connectivity Analysis procedure no longer returns a list of disconnected OD pairs, but rather a Numpy array
with the nodes that are not connected to the main portion of the network.
This change comes in the heels of replacing the previous Depth-First Search implementation that returned
a full connectivity matrix with the use of a *Connected Components* algorithm, which runs in a few seconds even
for very large networks and is not limited to centroid connectivity, which is substantially more useful for network
diagnostics and editing.

The connectivity analysis is also a feature of the AequilibraE graph, making it substantially easier to use.

.. testsetup::

    from pathlib import Path
    from tempfile import TemporaryDirectory

    from aequilibrae.utils.create_example import create_example

    _aequilibrae2_docs_tmp = TemporaryDirectory()
    project_path = str(Path(_aequilibrae2_docs_tmp.name) / "project")

.. doctest::

    >>> project = create_example(project_path)
    >>> project.network.build_graphs()

    # We get the graphs for cars and trucks
    >>> graph = project.network.graphs['c']
    >>> graph.disconnected_nodes()
    array([], dtype=int64)
