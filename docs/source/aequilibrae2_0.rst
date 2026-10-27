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


Public transport assignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optimal strategies kernel moved to C++. It keeps its scratch space between destinations and assignments,
takes new costs without rebuilding the graph, and loads each strategy without sorting it. On synthetic grids,
assignment takes about half the time it did:

+------------------------+--------------+---------+--------+--------+----------+
| Graph                  | Destinations | Threads | Before | 2.0    | Speed-up |
+========================+==============+=========+========+========+==========+
| 179,400 edges          | 1,000        | 1       | 10.6 s | 5.2 s  | 2.1x     |
+------------------------+--------------+---------+--------+--------+----------+
| 179,400 edges, 3 skims | 1,000        | 1       | 11.8 s | 5.7 s  | 2.1x     |
+------------------------+--------------+---------+--------+--------+----------+
| 179,400 edges          | 20,000       | 24      | 20.8 s | 11.2 s | 1.9x     |
+------------------------+--------------+---------+--------+--------+----------+
| 319,200 edges          | 4,000        | 24      | 8.2 s  | 4.1 s  | 2.0x     |
+------------------------+--------------+---------+--------+--------+----------+
| 319,200 edges, 3 skims | 4,000        | 24      | 9.1 s  | 4.8 s  | 1.9x     |
+------------------------+--------------+---------+--------+--------+----------+


API Changes
--------------------

.. toctree::
   :maxdepth: 1

   table_api_migration
   matrix_api_migration
   transit_api_migration

Changes in Behaviour
--------------------

Public transport
~~~~~~~~~~~~~~~~
Transit graph building, assignment and skimming were repaired, and these repairs change results for
existing models. Saved transit graphs are dropped on upgrade and must be rebuilt, and GTFS feeds
imported before 2.0 keep the defects of the old importer until they are imported again. See
:doc:`transit_api_migration` for the API changes.

- **Line frequencies** are each line's distinct departures in the period over the period's length, rather than
  the inverse of a minimum gap between stop events, which could fall to zero and remove the wait altogether.
- **Default wait** is half a headway (``TransitCosting.headway_fraction = 0.5``), rather than a full headway.
- **Waiting time skims** are the expected waits of the strategy, rather than travel time minus in-vehicle, access and
  egress times, which also swept in alighting penalties, dwell and walking times.
- **Transfer skims** are expected boardings minus the probability of boarding at all, rather than counts of transfer
  edges, which were zero with the default settings.
- **Boardings** count every boarding, now that transfers alight to the stop and board again.
- **Unreachable pairs** skim as ``inf`` rather than ``0.0``, and their demand is reported as unassigned.
- **Connectors** are measured geodesically, rather than in Web Mercator, and every zone connects to at least its
  nearest stop.
- **Dwell** is the scheduled dwell at each stop, rather than a uniform 30 s added to times that already included it.
- **Post-midnight trips** belong to the whole-day default period, which now runs to ``48:00:00``.
- **Trips** belong to a period by a trip-level rule, by default their first departure, so run times and headways
  come from complete trips.
- **Zone order** no longer matters: demand and skims follow the graph's centroids whatever the order of the
  ``zones`` table.
- **Public transport preloads** use half-open windows, so a trip leaving exactly at the end of a window belongs to the
  next one.

On the Coquimbo example, with default settings except ``with_walking_edges=False`` and one trip between every
pair of zones, results before these repairs and in 2.0 compare as follows. The 07:00-09:00 rows average over the
380 pairs both versions reach.

+-------------+--------------------------------+--------------+---------------------------------+
| Period      | Result                         | Before       | 2.0                             |
+=============+================================+==============+=================================+
| Whole day   | Zones with a connector         | 20 of 133    | 133 of 133                      |
+-------------+--------------------------------+--------------+---------------------------------+
| Whole day   | Zone pairs reachable           | 380          | 17,556                          |
+-------------+--------------------------------+--------------+---------------------------------+
| Whole day   | Unreachable pairs skimmed as 0 | 17,176       | 0 (they skim as inf)            |
+-------------+--------------------------------+--------------+---------------------------------+
| Whole day   | Mean boarding frequency        | 12.0 trips/h | 7.5 trips/h                     |
+-------------+--------------------------------+--------------+---------------------------------+
| 07:00-09:00 | Mean in-vehicle time           | 2,477 s      | 2,066 s                         |
+-------------+--------------------------------+--------------+---------------------------------+
| 07:00-09:00 | Mean waiting time              | 968 s        | 178 s (357 s at a full headway) |
+-------------+--------------------------------+--------------+---------------------------------+
| 07:00-09:00 | Mean expected cost (trav_time) | 4,191 s      | 3,453 s                         |
+-------------+--------------------------------+--------------+---------------------------------+
| 07:00-09:00 | Mean transfers                 | 0.00         | 0.19                            |
+-------------+--------------------------------+--------------+---------------------------------+
| 07:00-08:00 | Bus preload (PCE x links)      | 12,484       | 12,288                          |
+-------------+--------------------------------+--------------+---------------------------------+

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

    # We build the graph for cars
    >>> graph = project.network.build_graph("c", "free_flow_time")
    >>> graph.disconnected_nodes()
    array([], dtype=int64)
    >>> project.close()
