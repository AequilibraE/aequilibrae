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


API Changes
--------------------

.. toctree::
   :maxdepth: 1

   table_api_migration
   matrix_api_migration

Changes in Behaviour
--------------------

GTFS import
~~~~~~~~~~~
The behaviour of GTFS importer changed in AequilibraE 2.0. Models created with previous versions, particularly those
that imported GTFS feeds that include ``frequencies.txt``, loop routes, scheduled dwell, consecutive stops with
equal times, free fares or unlimited transfers would benefit from being imported again.

Changes in behaviour and corrections include:

- **Scheduled dwell** is kept: ``trips_schedule`` stores each stop's arrival and departure as published, rather than
  the later of the two for both, which added the dwell at each stop to the run time towards it.
- **Consecutive stops with equal times** no longer stop the import: run times of zero become one second, and negative
  run times are repaired the same way.
- **Trips crossing midnight** in clock time (``23:58:00``, then ``00:02:00``) continue past ``24:00:00`` (``24:02:00``),
  rather than going back in time. Times written past ``24:00:00``, as GTFS specifies, are kept as published.

- **Maximum speeds** set with ``set_maximum_speeds`` are applied. Distances were measured in degrees, so they never
  were. Distances are in metres and speeds in metres per second.

- **frequencies.txt** trips are imported, one trip per departure from ``start_time`` up to but excluding ``end_time``,
  in every window of the trip. Every trip defined by frequencies was skipped before.

- **Loop routes** keep the pairs of stops they serve more than once, in ``trips_schedule`` and ``route_links``.

- **Patterns** are told apart by delimited stop IDs, so stop sequences such as ``1, 23`` and ``12, 3`` on one route are
  no longer the same pattern. Pattern IDs of routes with several patterns may differ from those of earlier imports.

- **Extended route types** (``100`` to ``1700``) take the default capacities, PCEs and maximum speeds of their basic
  type, and map matching a basic type also matches its extended types, so extended bus types such as ``700`` can be
  map matched.

- **Fares** are stored as published: a free fare keeps its price of zero, empty ``transfers`` (unlimited transfers),
  ``transfer_duration`` (no time limit) and ``payment_method`` are stored as NULL rather than zero, and fare zone IDs
  are text. Transit migration 008 declares ``fare_rules.contains`` as text, though zone IDs that earlier imports
  turned into numbers (such as ``07`` into ``7``) cannot be recovered.

The GTFS export code was also overhauled.

``Transit.export_gtfs`` writes the transit database to a zipped GTFS feed, with the GTFS IDs of the imported feeds.
GTFS requires the URL and time zone of the agencies, which import does not keep, so they are arguments. Each agency
gets its imported service date as its only service, in ``calendar_dates.txt``, and trips imported from
``frequencies.txt`` are exported as individual trips.

.. code-block:: python

    transit.export_gtfs("feed.zip", agency_url="https://www.agency.com", agency_timezone="America/Santiago")

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
