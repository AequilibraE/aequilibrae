# AequilibraE Project Context

Generated from a repository scan on 2026-05-21. Treat this as a brownfield orientation document for OpenSpec planning. It describes the current implementation shape; durable behavioral requirements should live in `openspec/specs/`.

## Related Documents

- `openspec/project.md` is the most detailed project-context snapshot for agents and spec planning.
- `openspec/specs/*/spec.md` files are the canonical current-state behavioral specs, organized by capability.
- `AGENTS.md` is the operational contract for Codex, GitHub Copilot, and other coding agents.
- `README.md` is the public package overview and should stay concise.

If these documents disagree, resolve the disagreement against implementation files, SQL schema, tests, and CI before updating the docs.

## Purpose

AequilibraE is a Python transportation modeling package. It supports project-based network modeling, traffic assignment, transit and GTFS workflows, trip distribution, matrix handling, and geospatial import/export. The codebase combines Python orchestration with Cython kernels for performance-critical network and matrix operations.

## Current Scale

- Main package: about 150 Python files under `aequilibrae/`.
- Compiled code: about 27 Cython-related files (`.pyx`, `.pxd`, `.pxi`).
- Database specification: about 43 SQL files under `aequilibrae/project/database_specification/`.
- Tests: about 85 `test_*.py` files under `tests/`.
- Documentation: Sphinx/reST docs, gallery examples, doctests, static images, and documentation deployment workflow.
- OpenSpec: initialized under `openspec/` with `schema: spec-driven`.

## Technical Stack

- Python: package requires Python 3.10+ and CI tests 3.10 through 3.14.
- Native build: setuptools, Cython, C++17, OpenMP.
- Data/science libraries: NumPy, SciPy, pandas, GeoPandas, Shapely, pyproj, rtree, pyarrow.
- Matrix format support: native `.aem`, OMX through `openmatrix`, sparse matrices through SciPy and Cython wrappers.
- Geospatial database: SQLite plus SpatiaLite; bundled blank `spatialite.sqlite` is copied into new projects.
- HTTP/network access: `requests` for OSM/Nominatim/Overpass, `urllib.request.urlretrieve` for Windows SpatiaLite support and some docs examples.
- Documentation: Sphinx, pydata-sphinx-theme, sphinx-gallery, doctest, LaTeX/PDF generation.
- Testing: pytest, pytest-xdist, pytest-random-order, pytest-cov, pytest-subtests.
- Linting: ruff 0.14.8, line length 120, rules B/C/E/F/W with B017 ignored.
- Packaging: cibuildwheel for CPython 3.10-3.14 wheels. The active wheel workflow builds on Ubuntu, Windows, and Ubuntu ARM. Old Python, 32-bit Windows, i686 Linux, musllinux, s390x, ppc64le, and macOS wheel tags are skipped by the current configuration/workflow.

## Repository Layout

- `aequilibrae/__init__.py` exposes the public package facade: `Project`, `Graph`, `TrafficAssignment`, `TrafficClass`, `AequilibraeMatrix`, distribution models, result holders, logging, parameters, and subpackages.
- `aequilibrae/context.py` stores the active `Project` singleton used by components that default to the currently open project.
- `aequilibrae/log.py` defines the global `aequilibrae` logger and project log file access.
- `aequilibrae/parameters.py` loads YAML parameters from the active project or from the package default `parameters.yml`.
- `aequilibrae/project/` owns project lifecycle, database connections, schema creation, migrations, scenarios, matrices, results, zoning, and network APIs.
- `aequilibrae/project/database_specification/` is the schema source of truth for project and transit databases.
- `aequilibrae/project/network/` contains link/node/mode/link-type/period APIs plus OSM and GMNS builders/exporters.
- `aequilibrae/paths/` contains graph construction, path computation, skimming, all-or-nothing loading, traffic assignment, transit assignment, route choice, connectivity, and result objects.
- `aequilibrae/paths/cython/` contains shortest path, graph compression, public transport, VDF, route choice, and path-file kernels.
- `aequilibrae/distribution/` contains gravity application/calibration, synthetic gravity model, and IPF.
- `aequilibrae/matrix/` contains the native matrix implementation and sparse/COO Cython helpers.
- `aequilibrae/transit/` contains GTFS import, route-system reading/writing, transit elements, route map matching, transit graph building, and preloading.
- `aequilibrae/utils/` contains database/geospatial helpers, SpatiaLite utilities, example creation, Delaunay network creation, signals, QGIS adapter stubs, and SimWrapper export.
- `aequilibrae/reference_files/` contains bundled blank/sample data files used to create examples and projects.
- `tests/` mirrors major package areas and includes test project data, GTFS data, migration fixtures, shapefiles, SQLite projects, and path-file fixtures.
- `docs/` contains Sphinx configuration, examples, generated table documentation helpers, website deployment helpers, and many static images.
- `benchmarking/` contains standalone performance scripts for transit and select-link workflows.

## Runtime Architecture

### Project Lifecycle

`Project` is the central runtime object. `Project.new(path)` creates a project folder by copying the bundled SpatiaLite database, writing `parameters.yml`, creating a `run/__init__.py`, and initializing network tables and triggers. `Project.open(path)` loads an existing folder containing `project_database.sqlite`, activates the project globally, attaches a project logger, and loads child gateways.

A loaded project owns:

- `scenario`: current `Scenario` object.
- `network`: network gateway and graph builder.
- `transit`: public transport gateway backed by `public_transport.sqlite`.
- `matrices`: matrix registry and matrix-file gateway.
- `results`: result registry backed by `results_database.sqlite`.
- `about`: project metadata gateway.
- `zoning`: zone and centroid tooling.

The project uses context managers for database access:

- `db_connection`: non-spatial connection to `project_database.sqlite`.
- `db_connection_spatial`: SpatiaLite connection to `project_database.sqlite`.
- `transit_connection`: SpatiaLite connection to `public_transport.sqlite`.
- `results_connection`: SQLite connection to `results_database.sqlite`.

`Project.close()` commits/cleans, closes log handlers, and deactivates the active project.

### Scenario Model

The root project lives in the project folder. Additional scenarios live under `scenarios/<scenario_name>/` and can be empty or cloned from the active scenario. Scenario creation copies databases and parameters as needed, then registers the scenario in the root project database.

### Parameter Model

Parameters are YAML-backed. `Parameters()` reads the active project's `parameters.yml` when one exists, otherwise it deep-copies the package default. Parameters cover run entry points, assignment equilibrium settings, distribution settings, network field definitions, OSM behavior, GMNS mappings, and system settings.

### Database Model

The project has three database/file layers:

- `project_database.sqlite`: core network, metadata, matrices registry, results registry, periods, scenarios, and transit graph configurations.
- `public_transport.sqlite`: GTFS-derived transit data, transit network tables, transit graph data, and transit migrations.
- `results_database.sqlite`: output tables registered through the project `results` table.
- `matrices/`: matrix files (`.omx`, `.aem`) registered in the project database.

Schema creation is driven by ordered `table_list.txt` files and SQL table files. Trigger creation is driven by `triggers_list.txt`. SQL files are split on `--#` and executed command by command for easier failure diagnosis.

Migrations are listed in Python `migrations.py` files and may be SQL or Python. The migration system tracks `MISSING`, `SKIPPED`, and `APPLIED` states in a `migrations` table. Python migrations expose a callable `migrate(...)` and run inside manual transactions using `AequilibraEConnection`.

### Network Data Model

Core network tables include:

- `links`: directed/undirected link records with `link_id`, `a_node`, `b_node`, `direction`, `distance`, `modes`, `link_type`, name, directional speed/travel-time/capacity fields, and SpatiaLite `LINESTRING` geometry.
- `nodes`: node records with `node_id`, `is_centroid`, `modes`, `link_types`, and SpatiaLite `POINT` geometry.
- `zones`: TAZ records with `zone_id`, `area`, `name`, and SpatiaLite `MULTIPOLYGON` geometry.
- `modes`: single-letter mode IDs with passenger-car equivalent, value of time, and persons-per-vehicle fields.
- `link_types`: facility/link-type metadata.
- `periods`: time periods, defaulting to period 1 for the whole day.
- `matrices`: matrix registry.
- `results`: result registry.
- `transit_graph_configs`: saved transit graph configurations.
- `scenarios`: registered scenarios.
- `attributes_documentation`, `about`, and `migrations`: metadata and schema support.

Triggers enforce much of the data integrity for spatial/network editing, including derived geometry fields, link/node consistency, mode/link-type consistency, zone area, periods, scenarios, and transit tables. Do not treat SQL table definitions alone as the full data contract.

### Network API

`Network` owns `Modes`, `LinkTypes`, `Links`, `Nodes`, and `Periods`. It can:

- import OSM networks through Nominatim/Overpass, gridding large query areas when necessary;
- import/export GMNS files;
- build mode-specific `Graph` instances from database links;
- compute network extent and convex hull;
- list modes and skimmable fields.

`Links`, `Nodes`, `Modes`, `LinkTypes`, and `Periods` use table-gateway patterns: load structure through `TableLoader`, instantiate record objects on demand, cache edits in memory, and save through record APIs.

### Graph And Assignment Architecture

`Graph` transforms a link DataFrame into a directed graph suitable for shortest-path computation. It normalizes column names to lower case, requires `link_id`, `a_node`, `b_node`, and `direction`, expands two-way links into directed edges, maps arbitrary node IDs to dense indices, places centroids first, builds forward-star arrays, and stores graph fields in pandas/NumPy structures.

Graph compression has two layers:

- dead-end removal;
- link contraction for degree-two sequences.

Compressed graph structures are built by Cython code and are used for assignment and skimming. Centroids are preserved, and centroid-through-flow blocking is supported.

Traffic assignment is organized through:

- `TrafficClass`: one demand class with graph and matrix.
- `TrafficAssignment`: multi-class assignment controller.
- `LinearApproximation`: equilibrium algorithm implementation.
- VDF implementations in Cython/Python wrappers: BPR, BPR2, conical, INRETS, Akcelik.
- result holders under `aequilibrae/paths/results/`.

Available assignment concepts include all-or-nothing, MSA, Frank-Wolfe variants, skimming, select-link outputs, path-file saving to Feather/Parquet, and multi-threading.

Transit assignment uses `TransitClass`, `TransitAssignment`, `OptimalStrategies`, and Cython public transport/hyperpath logic.

### Matrix Architecture

`AequilibraeMatrix` supports memory-only matrices and file-backed matrices. Native `.aem` files use a binary layout with a fixed header, core metadata, index metadata, and matrix blocks. OMX support is provided through `openmatrix`. Computational views select cores used by assignment and skimming.

Cython sparse matrix helpers wrap C++ vectors and can convert to/from SciPy sparse matrices or OMX datasets.

### Transit Architecture

`Transit` ensures a `public_transport.sqlite` database exists when a project is loaded. It creates GTFS builders, imports route systems, creates/saves/removes/loads transit graphs by period, and can build transit preload vectors for traffic assignment.

Transit tables include agencies, fares, links, nodes, stops, stop connectors, routes, route links, pattern mapping, trips, trip schedules, modes, node types, zones, trigger settings, and migrations. Transit route geometries use SpatiaLite `MULTILINESTRING`.

GTFS import is represented through element classes (`Agency`, `Fare`, `Route`, `Pattern`, `Stop`, `Trip`, etc.), reader/writer modules, route map matching, and graph construction that links transit data to project network periods and zones.

## Build And Packaging

Editable development install:

```bash
pip install -e ".[dev]"
```

CI commonly uses:

```bash
uv pip install -e ".[dev]"
```

`setup.py` defines Cython extensions for path loading, IPF, public transport, route choice, graph building, sparse matrix, and COO demand code. Compilation uses C++17 and OpenMP. Environment flags:

- `AEQ_DEBUG`: adds `-O0` and debug symbols.
- `AEQ_ASAN`: enables AddressSanitizer and, outside Windows, UndefinedBehaviorSanitizer.

macOS CI uses Homebrew LLVM for OpenMP during tests. Linux CI installs SpatiaLite packages. Windows test setup downloads SpatiaLite/SQLite DLL support. macOS wheel builds are not active in the current wheel matrix, and macOS wheel tags are skipped in `pyproject.toml`.

## Testing

Run the full test suite:

```bash
pytest tests/ --durations=50 --dist=loadscope -n 4 --random-order --verbose
```

Run a quicker local pass:

```bash
pytest tests/ -x
```

Coverage workflow runs:

```bash
pytest tests/ --cov=aequilibrae --cov-branch --cov-report=term-missing --dist=loadscope --numprocesses=4 --random-order --verbose
```

Important test patterns:

- `tests/conftest.py` creates cached example projects for `sioux_falls`, `nauru`, `coquimbo`, scenario projects, no-trigger projects, and transit projects.
- Doctest fixtures in root `conftest.py` provide `Project`, `Transit`, `AequilibraeMatrix`, NumPy, pandas, `Path`, and helper project paths to documentation examples.
- `faulthandler.enable()` is active in tests to improve diagnostics for native-code failures.
- SQL table and trigger list consistency is tested in `tests/test_list_of_files.py`.
- Tests cover distribution, matrix, path/assignment, project/database/network, transit, logging, and utilities.

## Documentation

Docs are Sphinx-based under `docs/source/`, with autodoc, autosummary, napoleon, doctest, sphinx-gallery, sphinx-design, sphinx-git, and pydata-sphinx-theme. Gallery examples live under `docs/source/examples/`.

Documentation CI:

- validates version/documentation metadata;
- runs doctests across core modules and selected `.rst` files;
- builds LaTeX/PDF with `plot_gallery=False`;
- builds HTML with gallery;
- publishes to S3 when secrets are present.

Changing public APIs, examples, parameters, data model behavior, or algorithm outputs often requires docs updates and may require doctest updates.

## Programming Style And Conventions

- Public API classes commonly use Sphinx/reST docstrings with `:Arguments:` and `:Returns:`.
- The package facade imports many public classes at top level. Avoid adding imports that create circular dependencies.
- Project-aware components often default to `get_active_project(must_exist=False)` and fall back to default parameters/loggers.
- Database access should use context managers from `Project` or `commit_and_close`.
- User-provided SQL values should be parameterized. Table/field-name interpolation exists in controlled internal paths, but new user-driven SQL should not format raw user strings into SQL.
- Record/gateway classes frequently cache objects in private dictionaries and expose `reload`, `refresh`, `save`, `delete`, or `list` methods.
- pandas DataFrames are common at API boundaries; NumPy arrays and memoryviews are common at computational boundaries.
- Graph logic lowercases DataFrame columns through `Graph.__setattr__`.
- Warnings are used for non-fatal convergence, missing records, project upgrade caveats, and data-quality issues.
- Library code should use the configured logger rather than `print`.

## Security And Robustness Notes

- The project loads SQLite/SpatiaLite extensions. On Windows it may download SpatiaLite binaries into a temp directory or use `AEQ_SPATIALITE_DIR`.
- OSM import reaches external Nominatim/Overpass endpoints configured in `parameters.yml`.
- Graphs are saved/loaded with `pickle`; do not load graph files from untrusted sources without an explicit risk decision.
- Some existing code builds SQL dynamically for trusted table/field identifiers. New code that handles user input must validate identifiers or use parameterized SQL for values.
- Migrations are powerful and can span multiple databases. Python migrations must preserve transaction integrity and avoid `executescript` in Python migration bodies.
- CI deployment uses AWS secrets only in GitHub Actions guarded by secret presence checks.
- Test and docs examples may download external assets; isolate or mark such behavior when adding tests.

## OpenSpec Guidance For Future Work

Use this file for orientation only. `README.md` may summarize this context publicly, and `AGENTS.md` may turn it into agent instructions, but capability-level behavior should be captured in specs. Create or update capability specs when changing behavior in these areas:

- project lifecycle, scenarios, migrations, database schema, or triggers;
- network import/export, network editing, or graph-building behavior;
- path computation, graph compression, assignment algorithms, VDFs, route choice, or skimming;
- transit import, route matching, transit graph construction, or transit assignment;
- matrix binary format, OMX behavior, sparse matrix behavior, or matrix registry behavior;
- public API signatures, CLI entry points, examples, docs promises, or parameter semantics;
- security-relevant handling of SQL, files, downloads, extensions, or untrusted serialized data.

Current baseline specs:

- `openspec/specs/project-lifecycle/spec.md` - project creation, opening, scenarios, connections, and closure
- `openspec/specs/network-datamodel/spec.md` - network tables, modes, zones, triggers, and import/export boundaries
- `openspec/specs/graph-and-paths/spec.md` - graph preparation, centroids, compression, paths, and skims
- `openspec/specs/traffic-assignment/spec.md` - traffic classes, VDFs, algorithms, results, skims, and reports
- `openspec/specs/matrix-io/spec.md` - native matrices, OMX, computational views, project records, and sparse helpers
- `openspec/specs/transit-gtfs/spec.md` - transit database, GTFS import, transit graphs, preload, and assignment
- `openspec/specs/database-migrations/spec.md` - migration listing, status tracking, ordering, execution, and upgrades
- `openspec/specs/documentation-and-examples/spec.md` - Sphinx docs, API docstrings, doctests, gallery examples, and publishing

For each change, prefer a small capability-focused OpenSpec change over a broad rewrite. This codebase has many implicit contracts enforced by tests, SQL triggers, Cython memory layout, and docs examples.

## High-Risk Areas

- Cython kernels and OpenMP behavior.
- SpatiaLite loading and platform-specific database setup.
- SQL triggers and migrations.
- Graph compression and centroid mapping.
- Assignment convergence and numerical reproducibility.
- Matrix binary layout and OMX interoperability.
- Scenario cloning and multi-database consistency.
- GTFS import and transit graph persistence.
- Documentation doctests that execute real project workflows.
