Understanding the changes in AequilibraE 2.0
============================================

AequilibraE 2.0 is a substantially different software than the 1.X versions that preceded it.

While the change from 0.X to 1.X was mostly due to the maturing of the software feature set with the addition of
public transport modelling capabilities, the change from 1.X to 2.0 includes a wide range of breaking changes
in the API and changes in software behaviour, in addition to a series of new features and performance improvements.


New Algorithms and features
---------------------------


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

``aequilibrae.project.network``, ``aequilibrae.project.data`` and ``aequilibrae.project.tools`` no longer
re-export their implementation classes. Import ``MigrationManager`` and ``NetworkSimplifier`` from
``aequilibrae.project``. ``Matrices``, ``Results``, ``Scenario``, ``Tables``, ``Log`` and ``Transit`` are
project-owned objects, not public imports or objects users should construct. Obtain them through
``project.matrices``, ``project.results``, ``project.scenario``, ``project.tables``, ``project.log()`` and
``project.transit``. Create or select scenarios with
``project.create_empty_scenario()``, ``project.clone_scenario()`` and ``project.use_scenario()``.
Likewise, obtain table and record objects through the project rather than importing ``Links``, ``Nodes``,
``Zones``, ``Periods`` or their record classes. ``About``,
``FieldEditor`` and the project-table base classes are not public re-exports; use ``project.about`` and
the relevant table's field editor for ordinary workflows.

Logging and SpatiaLite helpers have public imports too:

.. code-block:: python

    import logging

    from aequilibrae.utils import basic_config, ensure_spatialite_binaries, set_known_spatialite_folder

    logger = logging.getLogger("aequilibrae")

Use the standard logger instead of the removed ``aequilibrae.global_logger`` export. Low-level Cython modules,
routing contexts and the shared ``aequilibrae._assignment`` infrastructure are implementation details, not
additional public import namespaces.

Project creation and lifetime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create or open a ready-to-use project with class methods. A context manager closes the project's connections,
including when an operation raises an exception:

.. code-block:: python

    from aequilibrae import Project

    with Project.create("/path/to/new_project") as project:
        links = project.tables.links.data

    with Project.open("/path/to/existing_project") as project:
        network = project.road.build_network()

Bare ``Project()``, ``Project.new()`` and ``Project.from_path()`` are not supported. Without a ``with`` block,
call ``project.close()`` explicitly. Opening a project is no longer a second initialisation step on an empty
object: replace ``project = Project(); project.open(path)`` with ``project = Project.open(path)``, and
``project = Project(); project.new(path)`` with ``project = Project.create(path)``.
``create_example()`` also returns an open project and can be used as a context manager.

Project tables, road networks and scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The database-backed ``Network`` gateway is named ``Tables`` and accessed through ``project.tables`` or
``project.scenario.tables``. The independent, in-memory ``Network`` retains its name. Obtain individual tables
through their owning project; table classes and extension infrastructure are not public re-exports.
``project.about``, ``project.matrices`` and ``project.results`` retain their accessors.
Road network construction belongs to ``project.road`` or ``project.scenario.road``: use
``build_network()`` to copy tables into an independent ``Network``, or ``build_graph()`` to build a graph directly.
These build methods are not available on ``Tables``.

The project object layout separates persistent data from routing inputs and prepared graphs:

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

``project.network`` is removed, not an alias for either ``tables`` or ``road``. In particular, there is no
project-owned road ``graphs`` dictionary: keep the graph returned by a build method yourself.
``project.tables`` and ``project.road`` follow the current scenario. A network or graph already built is an
independent snapshot; switching scenarios or editing tables does not update it. Build a new one to use the
changed data. ``project.scenario.tables`` and ``project.scenario.road`` expose the same split on a scenario.

This separation concerns the Python object layout, not a renaming of project files. ``Tables`` still accesses
``project_database.sqlite``; matrices remain in ``matrices/``, saved results in ``results_database.sqlite``,
and transit data in ``public_transport.sqlite``. Do not rename databases or create a ``road/`` directory for
these accessors. See :ref:`aeq_project_structure` for the on-disk project and scenario layout.

Independent routing and assignment objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A routing ``Network`` owns copies of the source link data and its configuration. It can be constructed from
in-memory data with ``Network(links, ...)``, without a project, or copied from project tables. A ``Graph`` is
built from that network for one mode and cost; it does not own database connections or retain the source
DataFrames. Source attributes belong to ``network.links``, while prepared graph data is organised under
``graph.routing``, ``graph.mapping`` and ``graph.skimming`` rather than ``graph.network`` or ``graph.graph``.
Keep the network as well as the graph if the workflow needs source attributes.

.. code-block:: python

    from aequilibrae.examples import create_example

    with create_example("/path/to/example_project") as project:
        links = project.tables.links.data
        network = project.road.build_network()
        graph = network.build_graph("c", cost="distance", skimming=["distance"])

    # The network and graph remain usable after the project is closed.
    path = graph.compute_path(1, 2)

Use ``project.road.build_graph("c", "distance")`` when only a graph is needed. Cost, skimming and routing
policies are supplied when building the graph, rather than configured through the old mutable graph-building
sequence. Network configuration fields and graph topology are fixed after construction; build a new object
for a different configuration. ``graph.set_costs()`` remains available to replace routing costs without
changing topology. See :doc:`path_computation/aequilibrae_graph` for graph construction and mappings.

Build the independent network explicitly before traffic assignment:

.. code-block:: python

    from aequilibrae.traffic import TrafficAssignment

    network = project.road.build_network()
    assignment = TrafficAssignment(network=network, project=project, **assignment_options)

``TrafficAssignment`` accepts only the built ``Network``, not ``Tables``. The explicit ``project`` argument
supplies project parameters and ownership for saving results. Results can also be saved to an explicitly supplied
project in the saving method. A ``TrafficClass`` is now a configuration independent of a graph, constructed
with ``mode``, ``cost`` and ``demand``. ``TrafficAssignment`` receives the network and classes and builds their
graphs internally; traffic-class results are available after execution through
``assignment.class_results[class_name]`` rather than ``traffic_class.results``. See
:doc:`traffic_assignment/assignment_procedures` for the complete assignment setup.

Transit graph builders remain distinct from prepared transit graphs. Obtain a builder with
``builder = project.transit.create_graph()`` and convert it with ``builder.to_transit_graph()``. The resulting
``aequilibrae.transit.TransitGraph`` is used with ``TransitClass`` and ``TransitAssignment``, not the road
``Graph``/``TrafficClass`` pair. Shared implementation infrastructure does not make these public types
interchangeable. See :doc:`public_transport/hyperpath_routing` for transit assignment.

Network import/export remains available through ``project.tables.importer.source()``, ``.osm()``, ``.overture()``
and ``.gmns()``, and ``project.tables.export_to_gmns()``. ``project.road.build_graph()`` is the graph-building
convenience method. GTFS still uses ``project.transit.new_gtfs_builder()``, ``load_date()``, optional map
matching and ``save_to_disk()``. Use ``project.transit.export_gtfs()`` for GTFS export. For a scenario without
a transit database, call ``project.scenario.create_transit_database()`` before accessing ``project.transit``.
To reload saved transit graphs, call ``project.transit.load(period_ids=[period_id])`` and retrieve the builder
from ``project.transit.graphs[period_id]`` rather than importing ``TransitGraphBuilder`` to call ``from_db()``.

Explicit project ownership and parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Library workflows no longer infer ownership from the active project. Pass a project or connection explicitly to
project-dependent operations, including result saving and network simplification. Database path/connection helpers
require an explicit project folder. The active-project context API and ``Project.activate()``/``deactivate()``
remain available for explicit use.

``Parameters()`` loads independent software defaults. Use ``project.project_parameters`` for the owning project's
parameters, or ``Parameters(path=project_folder)`` for an explicit folder. ``write_back()`` requires a folder and
raises ``ValueError`` when none was supplied. GMNS import/export through ``project.tables`` uses that project's
parameters.

Project run modules
~~~~~~~~~~~~~~~~~~~

Run-module functions must accept an explicit ``project`` argument. ``Project.run`` supplies it along with the
configured arguments; the CLI also supplies it when ``--no-defaults`` is selected. Update custom project
``run/__init__.py`` files accordingly:

.. code-block:: python

    def results_summary(project):
        return project.results.list()

Existing projects may be using outdated ``run/__init__.py`` files, check their
imports and function signatures and update them if appropriate. See the example
projects for the new sample file.  See :doc:`run_module`.

Removed compatibility entry points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Project.db_connection_spatial`` is removed; use ``Project.db_connection``. Deprecated trigger wrappers
``add_triggers`` and ``remove_triggers`` are removed. Internal callers suspending triggers use
``aequilibrae.utils.sqlite_utils.without_triggers`` within a managed transaction; it is not re-exported from
project creation. Compatibility modules ``distribution.ipf_core`` and ``paths.public_transport`` are removed.
``TransitGraphBuilder`` no longer accepts the unused ``seed``, ``geometry_noise`` or ``noise_coef`` arguments,
and its saving methods no longer accept ``robust``.

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
