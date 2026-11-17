Understanding the changes in AequilibraE 2.0
============================================

AequilibraE 2.0 is a substantially different software than the 1.X versions that preceded it.

While the change from 0.X to 1.X was mostly due to the maturing of the software feature set with the addition of
public transport modelling capabilities, the change from 1.X to 2.0 includes a wide range of breaking changes
in the API and changes in software behaviour, in addition to a series of new features and performance improvements.


New Algorithms and features
---------------------------

Routing and network workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Independent ``Network`` objects separate source data from routing graphs and remain usable after a project closes.
  Graphs have fixed topology, explicit network-to-graph mappings and costs that can be updated without rebuilding.
* Link contraction and dead-end removal can be selected independently. Path results can expand contracted paths
  back to physical link IDs and directions more easily.
* A* supports Euclidean and Haversine heuristics with explicit distance-to-cost scaling, including for non-distance
  costs. Priority queues are selectable, and connected-component diagnostics identify disconnected network nodes.

Turn restrictions and penalties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Mode-specific prohibited turns, turn penalties and configurable U-turn permission are supported in path finding,
  skimming, traffic assignment and route choice, including A* searches.
* Turn restrictions can be supplied in memory or edited through a dedicated project table. Database triggers validate
  movements and maintain their geometry as the network changes. Utilities list turns and find inapplicable restrictions.

Traffic assignment
~~~~~~~~~~~~~~~~~~

* Traffic classes are configured independently of graphs. Assignment builds each class's graph internally, allowing
  different mode-specific topologies while keeping congestion and results aligned to the same physical network.
* Improved conjugate and biconjugate Frank-Wolfe calculations now use total class flows with PCE weighting. BFW offers
  approximate and exact conjugacy options, additional diagnostics and safer fallbacks for unstable directions.
* Line search uses the Beckmann objective's directional derivative, including fixed and turn costs. Convergence
  reporting is aligned with the accepted flow, with improved handling of preloads and numerical edge cases.

Custom volume-delay functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* VDFs can be defined as Python functions or NumExpr expressions, or loaded from project parameters, without
  recompiling AequilibraE. Derivatives can be supplied explicitly or estimated numerically.
* Parameter specifications support bounds and missing-value defaults. A validation helper checks free-flow behaviour,
  monotonicity and convexity. Akcelik travel times and derivatives now consistently include link length.

Dense and sparse matrices
~~~~~~~~~~~~~~~~~~~~~~~~~

* Dictionary-like matrix collections separate in-memory calculations from OMX storage. Dense and sparse collections
  share zone indexing, metadata, named subsets and selection by matrix attributes.
* Label-based and positional indexers simplify matrix access. Constructors accept arrays, dictionaries and OD tables,
  including repeated OD pairs, while OMX stores support partial dense reads and writes.
* Sparse demand is supported in road assignment, transit assignment, route choice and sub-area analysis. Sparse OMX
  storage and sparse select-link OD outputs avoid unnecessarily creating dense matrices.
* Project matrix files are discovered directly, with metadata stored in OMX rather than a separate database catalogue.

Route choice and sub-area analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Link penalisation and BFS-LE can use A* as well as Dijkstra, with turn-aware route costs and path-size logit results.
* Route filtering uses an additive and multiplicative cost threshold relative to the cheapest route, replacing the
  probability-based cutoff. Imported routes can be validated and have costs, overlap and probabilities recomputed.
* Result-table construction is compiled to reduce Python overhead. Sub-area processing supports the new graph mappings
  and sparse demand while preserving original OD identifiers when requested.

Network import and simplification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* A shared importer supports OpenStreetMap downloads, local OSM PBF extracts and Overture Maps cloud data, alongside
  GMNS imports. Mode filtering covers cars, transit, bicycles and walking.
* OSM and Overture imports offer optional OSMnx or neatnet simplification and configurable intersection consolidation.
  Unsimplified networks are retained by default.
* Downloaded source data is cached with manifests, import provenance is recorded, and additional source attributes
  are preserved as JSON. Import and simplification benchmarks document performance and trade-offs.

Project tables and database editing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* A consistent table API provides immutable records, direct insert/update/delete methods and atomic bulk operations
  for network tables, zones, periods, turns and result metadata. Spatial tables accept Shapely geometry.
* Nested transactions use savepoints, allowing a failed batch to roll back without discarding unrelated outer edits.
  Database migrations use the same managed connections.
* Network triggers improve endpoint lookup and cascading updates, keeping link geometry, node attributes and turn
  restrictions consistent during editing. Trigger suspension is available within managed transactions.

Project organisation and saved results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Projects support context-managed opening and creation. Explicit project ownership and parameter selection make
  multi-project workflows less dependent on global state, including project run modules and CLI execution.
* Public imports are organised by workflow. ``project.tables`` handles database editing, ``project.road`` builds road
  networks and graphs, and ``project.transit`` provides transit workflows and builders.
* Result data and metadata are kept together in the results database. Creation and deletion manage both, named
  DataFrame index columns are preserved, and reconciliation helpers repair stale or missing metadata entries.

GTFS and public transport
~~~~~~~~~~~~~~~~~~~~~~~~~

* GTFS import handles frequency-based services, extended route types, repeated stops on loop routes and parent stations.
  Scheduled dwell, midnight crossings, Unicode text and published fare values are preserved more accurately.
* Malformed or incomplete feed records receive more robust handling and warnings. Maximum-speed checks use consistent
  distance units, and map matching handles extended route types and reverse-direction searches correctly.
* GTFS export writes zipped feeds using imported identifiers, with optional prefixes to distinguish agencies and dates.
  It retains parent stations and exports frequency-based services as individual trips.

Runtime performance and feedback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Routing, skimming, all-or-nothing loading, select-link processing and path-file writing use reworked Cython/C++
  kernels and OpenMP rather than Python thread pools or multiprocessing, with improved buffer reuse and memory locality.
* Separate thread settings and workload thresholds reduce overhead for small, memory-bound array operations.
* Standard Python logging supports scenario-aware file output and progress-friendly terminal output. Compiled workers
  can report progress and logs safely, and route choice and transit assignment support interruption.

Development and documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Meson-Python replaces setuptools, with a ``src`` package layout and incremental native builds. Spin provides shared
  build, test, lint, documentation and debugging commands.
* Expanded regression tests, assignment validation, benchmarks and native-code checks improve coverage of numerical
  correctness and performance. Updated examples, API migration guides, internals and profiling guides document 2.0.
* DataFrame operations have been updated for pandas copy-on-write behaviour.


Performance improvements
------------------------

Network skimming
~~~~~~~~~~~~~~~~

The parallelisation of the network skimming procedure was re-implemented in Cython, bringing tangible performance
improvements for small and medium networks, which cover a significant amount of the real-world networks
in use by transportation planning agencies in 2026.

TODO: RUN THESE BENCHMARKS

.. list-table:: Network skimming benchmarks
   :header-rows: 1

   * - Network
     - Links
     - Graph Links
     - Nodes
     - Centroids
     - AequilibraE 2.0 speed-up
   * - Arkansas
     - 300,146
     - 591,489
     - 243,059
     - 6,445
     - 0%
   * - Australia wide
     - 4,113,479
     - 7,760,714
     - 30,000,000
     - 60,000
     - 0%
   * - Chicago
     - 39,018
     - 39,018
     - 12,979
     - 1,790
     - 12%
   * - Coquimbo
     - 19,983
     - 34,546
     - 15,724
     - 133
     - 31%
   * - Lyon
     - 253,305
     - 468,761
     - 179,940
     - 4,617
     - 0%

The following conditions apply to this benchmark:

- The skimming procedure was run for a single mode and a single cost variable (i.e. "distance" for cars).
- It does not include the time taken to build the graph or to save the result to disk.
- The benchmarking was run on a desktop PC equipped with a intel i9 285K with 192GB of RAM, and
using 8 threads for the parallelisation (this CPU's *performance cores*).
- The speed-up procedure considers the ratio between the median time for 10 runs.


API Changes
--------------------

Public imports and package structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each public object has one documented import namespace. Import the objects used by a workflow from that
namespace rather than from the package root or an implementation module:

.. list-table:: Public namespaces
   :header-rows: 1
   :widths: 20 80

   * - Namespace
     - Public objects
   * - ``aequilibrae``
     - ``Project``, ``__version__``
   * - ``aequilibrae.project``
     - ``Parameters``, ``MigrationManager``, ``NetworkSimplifier``
   * - ``aequilibrae.routing``
     - ``Network``, ``Graph``, ``PathResults``, ``SkimResults``, ``NetworkSkimming``,
       ``available_heaps``, ``estimate_heuristic_scale``
   * - ``aequilibrae.traffic``
     - ``TrafficClass``, ``TrafficAssignment``, ``AssignmentResults``, ``AssignmentPaths``, ``VDF``
   * - ``aequilibrae.traffic.vdf``
     - ``bpr``, ``bpr2``, ``conical``, ``inrets``, ``akcelik``, ``builtin_vdfs``
   * - ``aequilibrae.transit``
     - ``TransitClass``, ``TransitAssignment``, ``TransitGraph``, ``TransitAssignmentResults``
   * - ``aequilibrae.route_choice``
     - ``RouteChoice``, ``SubAreaAnalysis``
   * - ``aequilibrae.distribution``
     - ``Ipf``, ``GravityCalibration``, ``GravityApplication``, ``SyntheticGravityModel``
   * - ``aequilibrae.matrix``
     - ``AequilibraeMatrix``, ``AequilibraeSparseMatrix``, ``MatrixStore``, ``from_df``, ``from_dict``, ``from_file``
   * - ``aequilibrae.examples``
     - ``create_example``, ``list_examples``
   * - ``aequilibrae.utils``
     - ``basic_config``, ``ensure_spatialite_binaries``, ``set_known_spatialite_folder``

The ``aequilibrae.paths`` package is replaced by
``aequilibrae.routing``. Traffic assignment, transit assignment, and route
choice implementations live in their respective packages. The package version
is available as ``aequilibrae.__version__`` rather than
``aequilibrae.version``.

The main import replacements are:

.. list-table:: Import migration
   :header-rows: 1
   :widths: 50 50

   * - Previous import
     - Replacement
   * - ``from aequilibrae.project import Project``
     - ``from aequilibrae import Project``
   * - ``from aequilibrae import Parameters``
     - ``from aequilibrae.project import Parameters``
   * - ``from aequilibrae import Log`` or ``from aequilibrae.project import Log``
     - Use ``project.log()``
   * - ``from aequilibrae import Matrices`` or ``from aequilibrae.project import Matrices, Results``
     - Use ``project.matrices`` and ``project.results``; do not construct their managers directly
   * - ``from aequilibrae.project import Scenario``
     - Use ``project.scenario`` and the project's scenario-management methods
   * - ``from aequilibrae.project.network import Network`` or ``from aequilibrae.project import Tables``
     - Use ``project.tables`` for database access;
       ``from aequilibrae.routing import Network`` for an independent network
   * - ``from aequilibrae import Graph, PathResults, SkimResults, NetworkSkimming``
     - ``from aequilibrae.routing import Graph, PathResults, SkimResults, NetworkSkimming``
   * - ``from aequilibrae.paths import available_heaps, estimate_heuristic_scale``
     - ``from aequilibrae.routing import available_heaps, estimate_heuristic_scale``
   * - ``from aequilibrae import TrafficClass, TrafficAssignment, AssignmentResults, AssignmentPaths, VDF``
     - Import these names from ``aequilibrae.traffic``
   * - ``from aequilibrae.paths import TransitClass, TransitAssignment, TransitGraph, TransitAssignmentResults``
     - Import these names from ``aequilibrae.transit``
   * - ``from aequilibrae.paths import RouteChoice, SubAreaAnalysis``
     - ``from aequilibrae.route_choice import RouteChoice, SubAreaAnalysis``
   * - ``from aequilibrae.transit import Transit``
     - Use ``project.transit``; create its database first if needed
   * - ``from aequilibrae.transit import TransitGraphBuilder`` or
       ``from aequilibrae.builder import TransitGraphBuilder``
     - Use ``project.transit.create_graph()``; reload saved builders with ``project.transit.load()``
   * - Imports of ``Importer``, ``GMNSBuilder`` or ``GMNSExporter`` from ``aequilibrae.builder``
     - Use ``project.tables.importer`` and ``project.tables.export_to_gmns()``
   * - Imports of ``GTFSRouteSystemBuilder`` or ``export_gtfs`` from ``aequilibrae.builder``
     - Use ``project.transit.new_gtfs_builder()`` and ``project.transit.export_gtfs()``
   * - ``from aequilibrae.paths.vdf import BPR``
     - ``from aequilibrae.traffic.vdf import bpr``
   * - ``from aequilibrae import Ipf, GravityCalibration, GravityApplication, SyntheticGravityModel``
     - Import these names from ``aequilibrae.distribution``
   * - ``from aequilibrae import AequilibraeMatrix, AequilibraeSparseMatrix, MatrixStore``
     - Import these names from ``aequilibrae.matrix``
   * - ``from aequilibrae.utils.create_example import create_example, list_examples``
     - ``from aequilibrae.examples import create_example, list_examples``

Objects previously imported from ``aequilibrae.paths`` or its implementation
modules follow a similar split: path computation and skimming now come from
``routing``, traffic assignment from ``traffic``, public transport assignment
from ``transit``, and route choice from ``route_choice``.

Project-dependent objects should be obtained from the project rather than
constructed directly. Use ``project.tables.importer`` for network imports,
``project.tables.export_to_gmns()`` for GMNS export, and the methods on
``project.transit`` for GTFS import/export and transit graph
construction. Implementation-module imports are for internal code and isolated
tests, not ordinary user workflows.

Most common operations should be done through through the ``project``, for
example: ``project.about``, ``project.matrices``, ``project.results``,
``project.scenario``, ``project.tables``, ``project.log()`` and
``project.transit``. Create or select scenarios with
``project.create_empty_scenario()``, ``project.clone_scenario()`` and
``project.use_scenario()``.  Likewise, obtain table and record objects through
the project rather than importing ``Links``, ``Nodes``, ``Zones``, ``Periods``
or their record classes.

Logging and SpatiaLite helpers have public imports too:

.. code-block:: python

    import logging

    from aequilibrae.utils import basic_config, ensure_spatialite_binaries, set_known_spatialite_folder

    logger = logging.getLogger("aequilibrae")

Use the standard logger instead of the removed ``aequilibrae.global_logger``
export.

Project creation and lifetime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a project with ``Project.create()`` or open one with the constructor.
When opening multiple projects, the context manager should be used to close the
project's connections upon completion, and when an operation raises an
exception:

.. code-block:: python

    from aequilibrae import Project

    with Project.create("/path/to/new_project") as project:
        links = project.tables.links.data

    with Project("/path/to/existing_project") as project:
        network = project.road.build_network()

Without a ``with`` block, call ``project.close()`` explicitly. Opening a
project no longer requires two-stage initialisation. Replace ``project =
Project(); project.open(path)`` with ``project = Project(path)``, and
``project = Project(); project.new(path)`` with ``project =
Project.create(path)``.  ``create_example()`` also returns an open project and
can be used as a context manager.

Project tables, road networks and scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common database operations should be done through the ``Tables`` object and
accessed through ``project.tables`` or ``project.scenario.tables``. More common
and general attributes such as ``project.about``, ``project.matrices`` and
``project.results`` retain on the ``project``. ``project.results`` now uses the
results database exclusively, and ``project.matrices`` no longer stores a
``matrices`` table. Road network construction has been moved to
``project.road`` or ``project.scenario.road``: use ``build_network()`` to load
the applicable tables into an independent ``Network``, or ``build_graph()`` to
build a graph directly for more advanced usages. The ``road`` attribute
parallels the ``transit`` attribute for transit network building.

The new project object layout aims to separates data operations from routing
and internals:

.. code-block:: text

    project
    ├── tables                 Database-backed table access and editing
    │   ├── links, nodes, zones
    │   ├── modes, link_types, periods
    │   ├── turn_restrictions
    │   └── importer
    ├── road                   Builds independent road networks and graphs
    ├── transit                Transit data and graph-building workflows
    ├── matrices               Project matrix files
    ├── results                Saved results and their metadata
    └── about                  Project metadata

.. list-table:: Project accessor migration
   :header-rows: 1
   :widths: 45 55

   * - Previous access
     - Replacement
   * - ``project.network.links`` (and other network tables)
     - ``project.tables.links`` (and the corresponding table accessor)
   * - ``project.scenario.network``
     - ``project.scenario.tables`` for table access; ``project.scenario.road`` for road-network building
   * - ``project.network.build_graphs()`` followed by ``project.network.graphs[mode]``
     - ``graph = project.road.build_graph(mode, cost)``
   * - ``project.network.importer``
     - ``project.tables.importer``
   * - ``project.network.export_to_gmns(path)``
     - ``project.tables.export_to_gmns(path)``

``project.network`` has been removed, it's has been replaced by the ``tables``
and ``road`` attributes. In particular, there is no ``network.graphs``
dictionary. Users should keep the graph returned by a build method
themselves. In general, AequilibraE has moved to the internal ``Graph``
structure being an internal artefact rather than a user facing object.  A
network or graph that has already been built is an does not change when
switching scenarios or editing tables. Users should build a new one to using
the modified data.

Independent routing and assignment objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A routing ``Network`` owns copies of the source link data and its
configuration. It can be constructed from in-memory data with ``Network(links,
...)``, without a project. A ``Graph`` is built from that network for one mode
and cost. The graph no longer stores a similar dataframe to ``network.links``,
it stores a prepared graph data and is organised under ``graph.routing``,
``graph.mapping`` and ``graph.skimming`` rather than ``graph.network`` or
``graph.graph``.  Keep the network as well as the graph if the workflow needs
source attributes, using ``graph.mapping`` to map between the network and
prepared graph.

.. code-block:: python

    from aequilibrae.examples import create_example

    with create_example("/path/to/example_project") as project:
        links = project.tables.links.data
        network = project.road.build_network()
        graph = network.build_graph("c", cost="distance", skimming=["distance"])

    # The network and graph remain usable after the project is closed.
    path = graph.compute_path(1, 2)

Use ``project.road.build_graph("c", "distance")`` when a graph is needed, most
of the time the ``Network`` should be enough. Cost, skimming and routing
options can be supplied when building the graph or configuring a
``TrafficClass``, rather than configured through the old mutable graph-building
with a sequence of ``graph.set_`` calls. Network configuration fields and graph
topology are fixed after construction, you should build a new object for a
different configuration. For quick cost updates, use ``graph.set_costs()`` to
replace routing costs without changing topology. See
:doc:`path_computation/aequilibrae_graph` for graph construction and mappings.

Build the independent network explicitly before traffic assignment:

.. code-block:: python

    from aequilibrae.traffic import TrafficAssignment

    network = project.road.build_network()
    assignment = TrafficAssignment(network=network, project=project, **assignment_options)

``TrafficAssignment`` accepts a ``Network`` and will be the appropriate graphs
for each ``TrafficClass``. The explicit ``project`` argument supplies project
parameters and ownership for saving results. Results can also be saved to an
explicitly supplied project in the saving method. A ``TrafficClass`` is now a
configuration independent of a graph, constructed with ``mode``, ``cost`` and
``demand``. ``TrafficAssignment`` should receive the network and classes and
builds their graphs internally, traffic-class results are available after
execution through ``assignment.class_results[class_name]`` rather than
``traffic_class.results``. See :doc:`traffic_assignment/assignment_procedures`
for the complete assignment setup.

Transit graph builders remain distinct from prepared transit graphs. Obtain a
builder with ``builder = project.transit.create_graph()`` and convert it with
``builder.to_transit_graph()``. The resulting
``aequilibrae.transit.TransitGraph`` is used with ``TransitClass`` and
``TransitAssignment``, not the road ``Graph`` / ``TrafficClass`` pair. See
:doc:`public_transport/hyperpath_routing` for transit assignment.

Network import/export are available through
``project.tables.importer.source()``, ``.osm()``, ``.overture()`` and
``.gmns()``, and ``project.tables.export_to_gmns()``. GTFS still uses
``project.transit.new_gtfs_builder()``, ``load_date()``, optional map matching
and ``save_to_disk()``. Use ``project.transit.export_gtfs()`` for GTFS
export. For a scenario without a transit database, call
``project.scenario.create_transit_database()`` before accessing
``project.transit``.  To reload saved transit graphs, call
``project.transit.load(period_ids=[period_id])`` and retrieve the builder from
``project.transit.graphs[period_id]`` rather than importing
``TransitGraphBuilder`` to call ``from_db()``.

Explicit project ownership and parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Library workflows no longer infer ownership from the active project. Pass a
project or connection explicitly to project-dependent operations, including
result saving and network simplification. Database path/connection helpers
require an explicit project folder.

Use ``project.project_parameters`` for the project's parameters, or
``Parameters(path=project_folder)`` for an explicit folder. GMNS import/export
through ``project.tables`` uses that project's parameters.

Project run modules
~~~~~~~~~~~~~~~~~~~

Run-module functions must accept an explicit ``project``
argument. ``Project.run`` supplies it along with the configured arguments. The
CLI also supplies it when ``--no-defaults`` is selected. Update custom project
``run/__init__.py`` files accordingly:

.. code-block:: python

    def results_summary(project):
        return project.results.list()

Existing projects may be using outdated ``run/__init__.py`` files, check their
imports and function signatures and update them if appropriate. See the example
projects for the new sample file.  See :doc:`run_module`.

Removed compatibility entry points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Project.db_connection_spatial`` has been removed, use
``Project.db_connection`` instead. Deprecated trigger wrappers ``add_triggers``
and ``remove_triggers`` are removed. Callers wanting to suspend triggers should
use ``aequilibrae.utils.sqlite_utils.without_triggers`` within a managed
transaction. Compatibility modules ``distribution.ipf_core`` and
``paths.public_transport`` have been removed.  ``TransitGraphBuilder`` no
longer accepts the unused ``seed``, ``geometry_noise`` or ``noise_coef``
arguments, and its saving methods no longer accept ``robust``.

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
  run times are repaired the same way. The number of times repaired is logged.
- **Trips crossing midnight** in clock time (``23:58:00``, then ``00:02:00``) continue past ``24:00:00`` (``24:02:00``),
  rather than going back in time. Times written past ``24:00:00``, as GTFS specifies, are kept as published.

- **Maximum speeds** set with ``set_maximum_speeds`` are applied. Distances were measured in degrees, so they never
  were. Distances are in metres and speeds in metres per second.

- **frequencies.txt** trips are imported, one trip per departure from ``start_time`` up to but excluding ``end_time``,
  in every window of the trip. Every trip defined by frequencies was skipped before. Their trip IDs, such as
  ``F-25200`` for trip ``F`` leaving at 07:00, never replace a trip of the feed.

- **Loop routes** keep the pairs of stops they serve more than once, in ``trips_schedule`` and ``route_links``, and
  the links to a stop visited a second time follow the shape rather than a straight line.

- **Patterns** are told apart by delimited stop IDs, so stop sequences such as ``1, 23`` and ``12, 3`` on one route are
  no longer the same pattern. Pattern IDs of routes with several patterns may differ from those of earlier imports.

- **Extended route types** (``100`` to ``1700``) take the default capacities, PCEs and maximum speeds of their basic
  type, and map matching a basic type also matches its extended types, so extended bus types such as ``700`` can be
  map matched.

- **Fares** are stored as published: a free fare keeps its price of zero, empty ``transfers`` (unlimited transfers),
  ``transfer_duration`` (no time limit) and ``payment_method`` are stored as NULL rather than zero, and fare zone IDs
  are text. Transit migration 008 declares ``fare_rules.contains`` as text, though zone IDs that earlier imports
  turned into numbers (such as ``07`` into ``7``) cannot be recovered.

- **Non-ASCII characters** are kept. Every character outside ASCII was removed from the feed before, so
  ``Peñuelas`` was imported as ``Peuelas``, in names, IDs and fare zones alike. Feeds must be encoded in UTF-8, as
  GTFS requires.

- **Fare rules** for a route or fare absent from the feed are skipped with a warning, and those for routes without
  service on the date imported are not saved. A rule for an unknown route applied to every route, and one for an
  unknown fare stopped the import. ``fare_zones`` lists the zones of every rule and stop, and stops without a zone
  have a NULL ``transit_fare_zone``.

- **Stations** that stops have as their ``parent_station`` are saved with the stops, so ``parent_station`` no longer
  refers to a missing stop.

- **Malformed files** no longer stop the import: rows that leave out their trailing empty fields, and files with a
  single row that lack optional columns, are read. Stop times for stops absent from ``stops.txt`` are dropped, and
  trips without times at their first or last stop are skipped, both with a warning.

The GTFS export code was also overhauled.

``Transit.export_gtfs`` writes the transit database to a zipped GTFS feed, with the GTFS IDs of the imported feeds.
GTFS requires the URL and time zone of the agencies, which import does not keep, so they are arguments. Each agency
gets its imported service date as its only service, in ``calendar_dates.txt``, and trips imported from
``frequencies.txt`` are exported as individual trips. Agencies that share GTFS IDs can only be exported together
with ``prefix_ids=True``, which prefixes each ID with its agency (e.g. ``Metro:1``), and also with the service date
for an agency imported for several dates (e.g. ``Metro_20260105:1``). Stops keep their ``parent_station``, and
stations are written with ``location_type`` 1.

.. code-block:: python

    transit.export_gtfs("feed.zip", agency_url="https://www.agency.com", agency_timezone="America/Santiago")

Connectivity Analysis
~~~~~~~~~~~~~~~~~~~~~

The Connectivity Analysis procedure returns a Numpy array with the nodes that
are not connected to the main portion of the network.  It uses a *Connected
Components* algorithm, which runs in a few seconds even for very large networks
and is not limited to centroid connectivity, which is substantially more useful
for network diagnostics and editing.

The connectivity analysis is also a feature of the AequilibraE graph, making it substantially easier to use.

.. testsetup::

    from pathlib import Path
    from tempfile import TemporaryDirectory

    from aequilibrae.examples import create_example

    _aequilibrae2_docs_tmp = TemporaryDirectory()
    project_path = str(Path(_aequilibrae2_docs_tmp.name) / "project")

.. doctest::

    >>> project = create_example(project_path)
    >>> network = project.road.build_network()
    >>> graph = network.build_graph('c', cost='distance')

    # Check for disconnected nodes
    >>> graph.disconnected_nodes()
    array([], dtype=int64)

Akcelik derivative
~~~~~~~~~~~~~~~~~~
The derivative of the Akcelik volume-delay function now includes the link length, as the function itself does.
In 1.X it was computed as if every link had unit length, and it returned the free-flow time on links without flow.
Both errors distorted the search directions of the conjugate and biconjugate Frank-Wolfe algorithms, so
assignments that use the Akcelik function with these algorithms converge differently than they did in 1.X.
