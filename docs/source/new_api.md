# AequilibraE 2.0 public API: design guidelines


## 1. What is wrong today

- **Too many entry points.** `aequilibrae/__init__.py` re-exports 26 names and imports every subpackage when it loads.
  `TrafficAssignment` can be imported from `aequilibrae`, `aequilibrae.paths` and
  `aequilibrae.paths.traffic_assignment`, so nobody can tell which location is the real one.

- **Grouped by how the code works, not by what it does.** `aequilibrae.paths` holds traffic assignment, transit
  assignment, route choice, sub-area analysis, VDFs, skimming and path finding. They are together only because they
  all run shortest paths. A user who wants to assign traffic should not need to know that.

- **`project.network` does five jobs.** It does CRUD on seven tables, imports networks (OSM, Overture, GMNS), exports
  GMNS, builds and caches graphs (`network.graphs`, `set_time_field`), and computes summaries (`count_links`,
  `extent`, `convex_hull`).
- 
- **Hidden dependency on the active project.** Seven modules outside `aequilibrae.project` call
  `get_active_project()`: in distribution, assignment, route choice, sub-area, assignment paths and the GTFS importer.
  Their results depend on whichever project was opened last. They cannot be used without a project, and with
  scenarios it is unclear which database they will write to.

- **Half-built objects.** `Project()` exists before it is opened. `TrafficAssignment()` is unusable until a fixed
  sequence of `set_*` calls has run. `network.graphs` is empty until `build_graphs()` is called. `Transit.graphs` is a
  class attribute, so every `Transit` instance in the process shares one dict.

- **Duplicate concepts.** There are three transit graph types (`TransitGraph`, `NewTransitGraph`,
  `TransitGraphBuilder`) and two names for each VDF (`bpr` / `BPR`). The same concept goes by `matrix`, `demand`,
  `skim_fields`, `skimming` and `skimming_fields`, depending on the class.

## 2. Library namespaces

### 2.1 Target map

| Namespace | Purpose | Public names | Today |
|---|---|---|---|
| `aequilibrae` | Entry point | `Project`, `__version__` | 26 names, plus eager imports of every subpackage |
| `aequilibrae.builder` |Everything to build models. OSM/Overture/GMNS | `GTFSBuider`, `GTFSExporter`, `OSMBuilder`, `OvertureBuilder` | . |
| `aequilibrae.matrix` | Matrix containers and constructors | `AequilibraEMatrix`, `MatrixStore`| Same, plus `Sparse`, `COO` and `GeneralisedCOODemand` |
| `aequilibrae.paths` | Graphs, which would be used if using Aeq without a model | `NetworkData`, `Graph`, `PathResults`, `SkimResults`, `NetworkSkimming`, `available_heaps`, `estimate_heuristic_scale` | Also holds the traffic, transit and route choice models |
| `aequilibrae.traffic` | Static traffic assignment and related classes | `TrafficClass`, `TrafficAssignment`, `AssignmentResults`, `AssignmentPaths`, `VDF` | `aequilibrae.paths` |
| `aequilibrae.traffic.vdf` | Catalogue of built-in VDFs | `bpr`, `bpr2`, `conical`, `inrets`, `akcelik`, `builtin_vdfs` | `aequilibrae.paths.vdf`, with upper-case duplicates |
| `aequilibrae.transit` | Transit assignment | `TransitClass`, `TransitAssignment`, `TransitGraph`, `TransitAssignmentResults` | `aequilibrae.paths` (the models) and `aequilibrae.transit` (GTFS, graph builder) |
| `aequilibrae.routechoice` | Route choice and sub-area demand | `RouteChoice`, `SubAreaAnalysis` | `aequilibrae.paths` |
| `aequilibrae.distribution` | Trip distribution | `Ipf`, `GravityCalibration`, `GravityApplication`, `SyntheticGravityModel`, `ipc_core` | Unchanged; these already sit in the right place |
| `aequilibrae.project` | Project, version, parameters, scenarios, migration_manager | `Scenario`, `Parameters`, `Log`, the table classes, `Matrices`, `Results`, `MigrationManager` | Mixed with `Project` itself |
| `aequilibrae.examples` | Bundled example models | `create_example`, `list_examples` | `aequilibrae.utils.create_example`, used 84 times in the docs |
| `aequilibrae.utils` | The few user-facing helpers | `basic_config`, `ensure_spatialite_binaries`, `set_known_spatialite_folder` | Every helper, public or not |

The imports a user writes:

```python
from aequilibrae import Project
from aequilibrae.matrix import AequilibraEMatrix, MatrixStore
from aequilibrae.paths import Graph # Or equivalents
from aequilibrae.traffic import TrafficAssignment, TrafficClass
from aequilibrae.traffic.vdf import bpr
from aequilibrae.transit import TransitAssignment, TransitClass
from aequilibrae.routechoice import RouteChoice
from aequilibrae.distribution import Ipf, GravityApplication
```

### 2.2 Rules for `__init__.py`

1. **The top level exports only `Project` and `__version__`.** It does not import subpackages. Importing
   `aequilibrae` must not load Cython kernels, SpatiaLite or GeoPandas.
2. **`__all__` is the contract.** A name is public only if a public namespace lists it in `__all__`. The module paths
   underneath (`aequilibrae.paths.traffic_assignment` and the like) are implementation details. They can move without
   a deprecation, and the docs never reference them.
3. **Each name has exactly one public home.** No class is re-exported at two levels, and no function has two
   spellings (drop `BPR = bpr`). The `set_vdf` should be case-insensitive
4. **Namespaces are one level deep.** The only exception is a catalogue of interchangeable built-ins, such as
   `aequilibrae.traffic.vdf` (`bpr` would be unclear at the parent level), or perhaps `aequilibrae.builder.transit`
5. **Imports have no side effects.** No logging setup, no extension loading, no file access.
6. **Internals stay internal.** `AssignmentInputs`, `LinearApproximation`, `HyperpathGenerating`,
   `OptimalStrategies`, the Cython modules, the importer pipeline, `db_utils` and `spatialite_utils` (apart from the
   helpers listed above) are never exported.
7. **Public names are spelled as the brand is.** `AequilibraeMatrix` becomes `AequilibraEMatrix`, as in the
   proposed imports. (This is not a hard rule, however)

## 3. The `Project` object

### 3.1 Target shape

```text
project
├── tables             one table at a time: CRUD on stored records
│   ├── links, nodes, zones, modes, link_types, periods, turn_restrictions, about
│   └── transit        tables in public_transport.sqlite: agencies, routes, stops, trips, ...
├── matrices           registry of OMX matrices: records plus files
├── builder            Building procedures
├── results            registry of result tables: records plus the results database
├── paths              read side: stored tables -> in-memory model inputs
│   ├── network(...)   -> NetworkData
│   ├── transit(...)   -> transit network data for a period
│   └── saved_transit(period_id)
├── traffic            workflows that span several link-network tables
├── transit            workflows on the transit database
├── scenarios          list / use / create / clone; .active, .root
├── parameters         parameters.yml
├── log                the project log
├── run                entry points of the run module
├── connections        raw database connections (advanced)
└── transaction()      a transaction across every open connection
```

The project is opened with `Project.open(path)` or `Project.create(path)`, both class methods, and can be used as a
context manager. `close()` and `upgrade()` keep their current meaning. Everything under `project` reflects the active
scenario, as it does today.

## 4. Other rules

1. **A configuration is complete and immutable when it is created.** Class definitions such as `TrafficClass` are
   frozen dataclasses that check their values in `__post_init__`. To change one, use `dataclasses.replace`. Never use
   `set_*` methods.
3. **Inputs are in-memory objects.** A model takes `NetworkData`, `Graph`, `AequilibraEMatrix` or `DataFrame`. It
   never takes a project, a table name or a file path.
4. **The model owns its graphs.** Procedures with several classes take `NetworkData` and build one graph per class,
   from its mode, cost, centroids and skims. Tools that work on a single graph (`compute_path`, skimming,
   `RouteChoice`) take a `Graph`, because the user controls that graph directly.
5. **No global project.** It is the end of `get_active_project()
6. **`execute()` returns a result object.** The result objects are the ones to write to disk
7. **One word per concept, everywhere.**

| Concept | Use | Do not use |
|---|---|---|
| Run a procedure | `execute()` | `run()`, `fit()`, `compute()` (`compute_*` is reserved for queries on `Graph`) |
| Threads | `cores` | `threads`, `n_jobs` |
| Routing field | `cost` | `cost_field`, `graph_field` |
| Fields to skim | `skimming` | `skim_fields`, `skimming_fields` |
| Network mode | `mode` | `mode_id` (that is only a table column name) |
