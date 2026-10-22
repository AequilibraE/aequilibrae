# AequilibraE 2.0 public API: design guidelines

This document sets the rules for reorganising AequilibraE's public API. It describes the target and the reasoning
behind it. It does not change any code. Implementation PRs should cite the section they follow, and any change to a
rule should be made here first.


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

## 2. The functional model

Every public object can be described by **what it does** and **what it needs**. Those two answers decide where it
belongs. Everything else in this document follows from that.

### 2.1 Four layers

| Layer | Holds | Needs | Examples |
|---|---|---|---|
| **Containers** | Data | Nothing | `AequilibraEMatrix`, `MatrixStore` |
| **Representations** | In-memory supply and the queries that need no demand | Data handed to it | `NetworkData`, `Graph`, `compute_path`, skims, connectivity |
| **Models** | Procedures that combine supply and demand, or that produce demand | Containers and representations | `TrafficAssignment`, `TransitAssignment`, `RouteChoice`, `Ipf` |
| **Project** | Persistent storage, and the workflows that read or write it | Everything | `Project`, tables, network importers, GTFS import |

**The dependency rule:** a layer may import from the layers above it in this table, never from those below.
Only the project layer touches the databases or the project folder. No code outside the project layer imports
`aequilibrae.project` or `aequilibrae.context`.

This rule decides most placement questions on its own:

- A GTFS import writes to `public_transport.sqlite`. That makes it a project workflow, even though it is about
  transit.
- Traffic assignment only computes. That makes it a model, and it must work on a `NetworkData` built by hand with no
  project at all. (Tom's branch already supports this.)
- Saving results is something the project does with a model's output. A model does not save itself.

### 2.2 Which namespace: decide by the inputs

| The code needs... | It lives in |
|---|---|
| Neither a network nor demand | `aequilibrae.matrix` |
| A network but no demand: shortest path, skims, connectivity | `aequilibrae.paths`, as `Graph` methods |
| Demand or impedance only, with no network: IPF, gravity models | `aequilibrae.distribution` |
| A network and demand, with equilibrium on link costs | `aequilibrae.traffic` |
| Transit supply and demand | `aequilibrae.transit` |
| A network and demand, with route choice sets | `aequilibrae.routechoice` |
| To read or write the project's databases or files | An attribute of `project`, never a library namespace |

Two more rules come from the same idea:

- **A result type lives with the procedure that creates it.** `AssignmentResults` moves to `aequilibrae.traffic`,
  `TransitAssignmentResults` to `aequilibrae.transit`, and `SkimResults` and `PathResults` stay in `aequilibrae.paths`.
- **Implementation details never decide placement.** Traffic, transit and route choice all run Dijkstra. That
  shared machinery stays private inside `aequilibrae.paths`, while each public class sits with its domain.

## 3. Library namespaces

### 3.1 Target map

| Namespace | Purpose | Public names | Today |
|---|---|---|---|
| `aequilibrae` | Entry point | `Project`, `__version__` | 26 names, plus eager imports of every subpackage |
| `aequilibrae.matrix` | Matrix containers and constructors | `AequilibraEMatrix`, `MatrixStore`, `from_df`, `from_dict`, `from_file`, `from_scipy` | Same, plus `Sparse`, `COO` and `GeneralisedCOODemand` |
| `aequilibrae.paths` | Routing on the link network: data, graphs, queries without demand | `NetworkData`, `Graph`, `PathResults`, `SkimResults`, `NetworkSkimming`, `available_heaps`, `estimate_heuristic_scale` | Also holds the traffic, transit and route choice models |
| `aequilibrae.traffic` | Static traffic assignment | `TrafficClass`, `TrafficAssignment`, `AssignmentResults`, `AssignmentPaths`, `VDF` | `aequilibrae.paths` |
| `aequilibrae.traffic.vdf` | Catalogue of built-in VDFs | `bpr`, `bpr2`, `conical`, `inrets`, `akcelik`, `builtin_vdfs` | `aequilibrae.paths.vdf`, with upper-case duplicates |
| `aequilibrae.transit` | Transit assignment | `TransitClass`, `TransitAssignment`, `TransitGraph`, `TransitAssignmentResults` | `aequilibrae.paths` (the models) and `aequilibrae.transit` (GTFS, graph builder) |
| `aequilibrae.routechoice` | Route choice and sub-area demand | `RouteChoice`, `SubAreaAnalysis` | `aequilibrae.paths` |
| `aequilibrae.distribution` | Trip distribution | `Ipf`, `GravityCalibration`, `GravityApplication`, `SyntheticGravityModel` | Unchanged; these already sit in the right place |
| `aequilibrae.project` | Types of the objects a `Project` hands out, for annotations | `Scenario`, `Parameters`, `Log`, the table classes, `Matrices`, `Results`, `MigrationManager` | Mixed with `Project` itself |
| `aequilibrae.examples` | Bundled example models | `create_example`, `list_examples` | `aequilibrae.utils.create_example`, used 84 times in the docs |
| `aequilibrae.utils` | The few user-facing helpers | `basic_config`, `ensure_spatialite_binaries`, `set_known_spatialite_folder` | Every helper, public or not |

`aequilibrae.transit` changes meaning more than any other namespace. Today it is the GTFS and transit-supply
package. In the target it holds the transit *models*. Importing GTFS and building transit graphs are project
workflows (§4), reached through `project.transit` and `project.paths`.

The imports a user writes:

```python
from aequilibrae import Project
from aequilibrae.matrix import AequilibraEMatrix, MatrixStore
from aequilibrae.paths import Graph, NetworkData
from aequilibrae.traffic import TrafficAssignment, TrafficClass
from aequilibrae.traffic.vdf import bpr
from aequilibrae.transit import TransitAssignment, TransitClass
from aequilibrae.routechoice import RouteChoice
from aequilibrae.distribution import Ipf, GravityApplication
```

### 3.2 Rules for `__init__.py`

1. **The top level exports only `Project` and `__version__`.** It does not import subpackages. Importing
   `aequilibrae` must not load Cython kernels, SpatiaLite or GeoPandas.
2. **`__all__` is the contract.** A name is public only if a public namespace lists it in `__all__`. The module paths
   underneath (`aequilibrae.paths.traffic_assignment` and the like) are implementation details. They can move without
   a deprecation, and the docs never reference them.
3. **Each name has exactly one public home.** No class is re-exported at two levels, and no function has two
   spellings (drop `BPR = bpr`).
4. **Namespaces are one level deep.** The only exception is a catalogue of interchangeable built-ins, such as
   `aequilibrae.traffic.vdf`, where a short name like `bpr` would be unclear at the parent level.
5. **Imports have no side effects.** No logging setup, no extension loading, no file access.
6. **Internals stay internal.** `AssignmentInputs`, `LinearApproximation`, `HyperpathGenerating`,
   `OptimalStrategies`, the Cython modules, the importer pipeline, `db_utils` and `spatialite_utils` (apart from the
   helpers listed above) are never exported.
7. **Public names are spelled as the brand is.** `AequilibraeMatrix` becomes `AequilibraEMatrix`, as in the
   proposed imports. This is a rename, so it is listed in §8.

## 4. The `Project` object

### 4.1 Target shape

```text
project
├── tables             one table at a time: CRUD on stored records
│   ├── links, nodes, zones, modes, link_types, periods, turn_restrictions, about
│   └── transit        tables in public_transport.sqlite: agencies, routes, stops, trips, ...
├── matrices           registry of OMX matrices: records plus files
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

### 4.2 What goes where, and why

Each attribute is defined by one criterion. If an operation meets two criteria, take the first that matches, going
down this list.

| Attribute | Criterion | Consequence |
|---|---|---|
| `tables` | The rows *are* the data, and the operation touches one table. | The uniform `ProjectTable` API: `get`, `insert`, `update`, `delete`, the `*_from` bulk methods, `data`, `fields`, `len()`. Queries on a single table also belong here, such as `links.extent()`. |
| `matrices`, `results` | The rows *describe* artifacts stored elsewhere (OMX files, tables in the results database). | Registries create, list, load and delete the artifact and its record together. Every stored model output is saved through one of them. |
| `paths` | The operation reads stored tables and returns a new in-memory object for a model. | It builds explicitly on every call, with no cache. The objects it returns hold no reference to the project. |
| `traffic` | A workflow over the link network that spans several tables: import, export, connecting zones, simplifying. | No CRUD (that is `tables`) and no graph building (that is `paths`). |
| `transit` | A workflow on the transit database: GTFS import, map matching, saving transit graphs, the preload for traffic assignment. | Same split as `traffic`. Reading transit tables goes through `tables.transit`. |
| `scenarios`, `parameters`, `log`, `run`, `connections` | Housekeeping. | Grouped so that the top level stays short. |

Some consequences:

- **Every object has one path to it.** Links are reached at `project.tables.links` and nowhere else.
  `project.traffic` does not re-expose tables.
- **`project.paths.network` is a method, not a property.** It takes arguments (`fields`, `modes`, `limit_to_area`)
  and returns a fresh snapshot each time. A property would hide an expensive build and invite caching, and a cache
  goes stale as soon as the tables are edited. Tom's branch removes the `network.graphs` cache for the same reason.
- **Transit tables live under `project.tables.transit`.** The transit database reuses table names from the project
  database (`links`, `nodes`, `modes`, `link_types`, `zones`), so a flat `tables` namespace would have collisions.
  Nesting the transit tables also keeps "this project has no transit database" a single, clear error.
- **Reading belongs to `paths`, writing to the system that owns the database.** A saved transit graph is loaded
  with `project.paths.saved_transit(...)` and saved with `project.transit.save_graph(graph)`. The builder no longer
  saves itself, because the objects `paths` returns hold no reference to the project.
- **A `Project` object is always open.** With `Project.open` and `Project.create` as class methods, there is no
  "this project is not opened" state to check for.

### 4.3 Mapping from the current surface

The method names under `traffic` and `transit` are proposals. The placement is what this document fixes.

| Today | Target |
|---|---|
| `Project()` + `.new(path)` / `.open(path)`, `Project.from_path(path)` | `Project.create(path)`, `Project.open(path)` |
| `project.about` | `project.tables.about` |
| `project.network.{links, nodes, zones, modes, link_types, periods, turn_restrictions}` | `project.tables.{...}` |
| `project.network.build_graphs()` / Tom's `project.network.build()` | `project.paths.network(...)` |
| `project.network.graphs`, `project.network.set_time_field()` | Removed. Choose the cost when you build a graph: `network.build_graph(mode, cost)` |
| `project.network.skimmable_fields()` | `NetworkData.fields` |
| `project.network.list_modes()` | `list(project.tables.modes)` |
| `project.network.count_links()` / `count_nodes()` | `len(project.tables.links)` / `len(project.tables.nodes)` |
| `project.network.count_centroids()` | A centroid query on `project.tables.nodes` |
| `project.network.extent()` / `convex_hull()` | `project.tables.links.extent()` / `.convex_hull()` |
| `project.network.importer.{osm, overture, gmns, source}` | `project.traffic.importer.{...}` |
| `project.network.export_to_gmns(path)` | `project.traffic.export_gmns(path)` |
| `zones.connect_mode`, `zones.disconnect_mode`, `nodes.connect_mode`, `zones.add_centroid(s)` | `project.traffic.connect_zones(...)` and the related calls (each touches nodes, links and zones) |
| `NetworkSimplifier(project).simplify(graph)` / `.collapse_links_into_nodes()` | `project.traffic.simplify(...)` / `project.traffic.collapse_links(...)` |
| `project.transit.get_table(name)` | `project.tables.transit.<name>` |
| `project.transit.new_gtfs_builder(...)` | `project.transit.import_gtfs(...)` |
| `project.transit.create_graph(period_id=...)` | `project.paths.transit(period_id=..., ...)` |
| `project.transit.load(period_ids)` | `project.paths.saved_transit(period_id)` |
| `project.transit.save_graphs()` / `remove_graphs()` | `project.transit.save_graph(graph)` / `remove_graphs(period_ids)` |
| `project.transit.build_pt_preload(...)` | `project.transit.preload(...)` |
| `project.scenario.create_transit_database()` | `project.transit.create_database()` |
| `list_scenarios()`, `use_scenario()`, `create_empty_scenario()`, `clone_scenario()` | `project.scenarios.list()`, `.use()`, `.create()`, `.clone()` |
| `project.scenario`, `project.root_scenario` | `project.scenarios.active`, `project.scenarios.root` |
| `project.parameters` (a dict) and `project.project_parameters` (an object) | `project.parameters`, one object with dict access |
| `project.log()` | `project.log` |
| `project.project_base_path`, `project.path_to_file` | `project.path`; the file path becomes private |
| `db_connection`, `results_connection`, `transit_connection`, `db_connection_spatial` | `project.connections.{project, results, transit}`; the spatial alias is dropped |
| `activate()`, `deactivate()` | Private: opening a project activates it, for the run module |
| `check_file_indices()` | Removed (it raises `NotImplementedError`) |
| `AssignmentPaths(table_name, project=...)` | `project.results.load_paths(table_name)`, which returns `aequilibrae.traffic.AssignmentPaths` |
| `DelaunayAnalysis(project)` | Not decided yet (§8) |

## 5. The shape of a model

Tom's branch sets the pattern for traffic assignment. Every model follows it.

1. **A configuration is complete and immutable when it is created.** Class definitions such as `TrafficClass` are
   frozen dataclasses that check their values in `__post_init__`. To change one, use `dataclasses.replace`. Never use
   `set_*` methods.
2. **The constructor defines the problem. `execute()` decides how hard to solve it.** Network, classes, VDF, fields
   and algorithm go to the constructor, which checks them all eagerly. Iterations, gap, cores, threading and logging
   go to `execute()`. The same object can then run with different budgets.
3. **Inputs are in-memory objects.** A model takes `NetworkData`, `Graph`, `AequilibraEMatrix` or `DataFrame`. It
   never takes a project, a table name or a file path.
4. **The model owns its graphs.** Procedures with several classes take `NetworkData` and build one graph per class,
   from its mode, cost, centroids and skims. Tools that work on a single graph (`compute_path`, skimming,
   `RouteChoice`) take a `Graph`, because the user controls that graph directly.
5. **No `project=` parameter and no active-project lookup.** Defaults come from the library's default parameters.
   Pass `project.parameters[...]` explicitly when you want the project's values. The run module (`project.run`) is
   the only code allowed to call `get_active_project()`. This applies to `Parameters` too: `Parameters()` with no
   path must mean the library defaults. Today it silently reads the active project's `parameters.yml`.
6. **`execute()` returns a result object, and the model never writes to disk.** The result carries its outputs
   (link flows, skims, select-link outputs, convergence) and its metadata (procedure name, procedure id, report). The
   registries can then store it from a name alone, and the `save_*(..., project=None)` methods are removed.
7. **One word per concept, everywhere.**

| Concept | Use | Do not use |
|---|---|---|
| Run a procedure | `execute()` | `run()`, `fit()`, `compute()` (`compute_*` is reserved for queries on `Graph`) |
| Threads | `cores` | `threads`, `n_jobs` |
| OD input | `demand` | `matrix` |
| Routing field | `cost` | `cost_field`, `graph_field` |
| Fields to skim | `skimming` | `skim_fields`, `skimming_fields` |
| Network mode | `mode` | `mode_id` (that is only a table column name) |

### 5.1 End to end

The registry methods marked "illustrative" stand for rule 6. They are not an existing API.

```python
from aequilibrae import Project
from aequilibrae.traffic import TrafficAssignment, TrafficClass
from aequilibrae.traffic.vdf import bpr

with Project.open(path) as project:
    network = project.paths.network(modes=["c"])
    demand = project.matrices.get_matrix("demand_omx", subset=["matrix"])  # AequilibraEMatrix (#859)

    cars = TrafficClass(mode="c", cost="free_flow_time", demand=demand, skimming=("free_flow_time", "distance"))
    assignment = TrafficAssignment(
        network=network,
        classes=[cars],
        vdf=bpr,
        name_mapping={"alpha": "b", "beta": "power"},
        time_field="free_flow_time",
        capacity_field="capacity",
        algorithm="bfw",
    )
    result = assignment.execute(max_iterations=500, rgap=1e-5, cores=8)

    project.results.save("base_year", result)  # illustrative
    project.matrices.save("base_year_skims", result.skims["c"])  # illustrative

    # Queries without demand stay on the graph
    graph = network.build_graph(mode="c", cost="free_flow_time", skimming=["distance"])
    path = graph.compute_path(origin=1, destination=20)
```

## 6. Reconciling `tom/graph-and-assignment-interface-rewrite`

**Adopted as it is.** `NetworkData` as the snapshot from storage. The frozen `TrafficClass` holding mode, cost,
demand, PCE, VOT, fixed cost, select-link sets, `block_centroid_flows` and heap. The `TrafficAssignment` constructor
that takes the whole problem, with run options on `execute()`. Removing `network.graphs` and `set_time_field`.
Graphs that the assignment builds per class.

**To change before or while it lands:**

1. **Rebase on #859.** The branch predates the matrix rework. It still calls `demand.computational_view(...)`,
   `matrix_view` and `view_names`, which no longer exist. Demand becomes an in-memory `AequilibraEMatrix`, and every
   matrix in it is assigned as that class's demand. To pick matrices, use `project.matrices.get_matrix(..., subset=...)`
   or `.subset(...)`.
2. **`project.network.build()` becomes `project.paths.network()`.**
3. **Do not export `NetworkData` at the top level.** Its home is `aequilibrae.paths`, by rule 3.2.1.
4. **Remove `project=` from `TrafficAssignment` and `AssignmentBase`.** Read the defaults from the library's
   parameters (rule 5.5).
5. **Return a result object from `execute()`.** Today the branch returns a link `DataFrame` and keeps `class_results`,
   `report()`, `get_skim_results()`, `skim_congested()` and the `save_*` methods on the assignment. They move to the
   result object (rule 5.6).
6. **Give transit the same shape.** `TransitAssignment` still uses `set_classes`, `add_class`, `set_algorithm`,
   `set_cores`, `set_time_field` and `set_frequency_field`. The target is a frozen `TransitClass` plus
   `TransitAssignment(network=<transit network data>, classes=[...], time_field=..., frequency_field=..., algorithm=...)`.
   The assignment builds its own graph, as the traffic assignment does from `NetworkData`.
7. **Keep one transit graph type.** `NewTransitGraph`, which `TransitGraphBuilder.to_transit_graph()` produces,
   becomes `aequilibrae.transit.TransitGraph`. The `GraphBase`-derived `TransitGraph` is removed.

## 7. Migration

- **Make a clean break in 2.0.** The table API migration set the precedent: removed classes and methods get no
  compatibility aliases.
- **Point people to the new location.** For the 2.0.x series, each old namespace defines a module-level `__getattr__`
  that raises `ImportError` naming the new import, for example "`TrafficAssignment` moved to `aequilibrae.traffic`".
  These are pointers, not working aliases.
- **Write `docs/source/api_migration.rst`** in the style of `table_api_migration.rst` and `matrix_api_migration.rst`,
  using §4.3 as its backbone.
- **Enforce the rules with tests.** One test compares each public namespace's `__all__` against an approved list, so
  changes to the API are deliberate and reviewed. A second test checks the dependency rule in §2.1: no module outside
  `aequilibrae/project/` imports `aequilibrae.project` or `aequilibrae.context`. A short AST walk or an import-linter
  contract will do. To keep this a simple path rule, project-layer code (the GTFS importer, `TransitGraphBuilder`)
  eventually moves under `aequilibrae/project/`. That move does not change the public API, so it can come last.

Suggested order. Each step leaves the test suite green.

1. Land Tom's branch, rebased on #859, with items 6.1 to 6.3.
2. Create `aequilibrae.traffic`, `aequilibrae.routechoice`, `aequilibrae.transit` (models) and
   `aequilibrae.examples`. Move the classes and shrink every `__init__.py` to its §3.1 list.
3. Restructure `Project`: `tables`, `paths`, `traffic`, `transit`, `scenarios`, the class-method constructors.
4. Remove `project=` and `get_active_project()` from the models. Add result objects and registry saving (items 6.4
   and 6.5).
5. Give transit assignment the same shape (items 6.6 and 6.7).
6. Write the migration guide, update the examples, and add the API and layering tests.

## 8. Open questions

Each question has a recommendation. Once it is settled, update the relevant section and delete the question.

1. **Is `project.traffic` the right name?** The link network also carries the walk and bike modes that transit uses
   for access. `project.network` becomes free once graphs move to `paths`. *Recommendation:* keep `traffic`, so that
   it reads as the counterpart of `transit`, and document that it covers every mode on the link network.
2. **Rename `AequilibraeMatrix` to `AequilibraEMatrix`?** *Recommendation:* yes, in the same release as the namespace
   move, so users change their imports only once.
3. **Where does `DelaunayAnalysis` go?** Today it does three jobs. It triangulates the zone or centroid points, writes
   the result as a `delaunay_network` table in the project database, and assigns a matrix to it. *Recommendation:*
   split it along the layers. The triangulation becomes a model that turns points into a `NetworkData`, in
   `aequilibrae.paths`. The assignment is an ordinary `TrafficAssignment`. The network layer and the flows are saved
   through `project.results`. Once split, it needs no home of its own.
4. **Where do the remaining utilities go?** These are `find_non_applicable_turn_restrictions`, the SimWrapper
   config generator and the `SIGNAL` progress hooks used by the QGIS plugin. *Recommendation:* the turn-restriction
   check becomes a method on `NetworkData`. SimWrapper becomes `project.results.export_simwrapper(...)`. `SIGNAL`
   stays in `aequilibrae.utils` as a documented integration hook.
5. **Should the run module receive the project explicitly?** That would remove `get_active_project()` completely,
   but it breaks existing `run/__init__.py` files. *Recommendation:* not in 2.0. Keep the active project only for
   the run module (rule 5.5).
