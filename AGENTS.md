# AGENTS.md

## Big picture
- `aequilibrae` is a transportation modeling library built around a **project folder** whose source of truth is SQLite/SpatiaLite databases plus a per-project `parameters.yml`.
- The central entry point is `Project` (`aequilibrae/project/project.py`). Opening/creating a project wires up `network`, `matrices`, `results`, `transit`, `zoning`, and scenario support.
- Network editing and persistence live in the database layer (`aequilibrae/project/`, `aequilibrae/project/database_specification/`). Runtime algorithms work mostly on pandas/NumPy/Cython objects built from that DB.
- Assignment flow is: `Project.network.build_graphs()` -> mode-specific `Graph` objects (`aequilibrae/paths/graph.py`) -> `TrafficClass`/`TransitClass` (`aequilibrae/paths/traffic_class.py`) -> `TrafficAssignment`/`TransitAssignment` (`aequilibrae/paths/traffic_assignment.py`).
- `Graph.prepare_graph()` builds a directed graph, reindexes centroids first, and creates a compressed graph for assignment/skimming. Many path algorithms assume this prepared/compressed representation exists.
- Matrices are first-class objects: `AequilibraeMatrix` supports native `.aem` and OMX. Assignment code expects `matrix.computational_view(...)` to be set and matrix indices to match `graph.centroids` exactly.

## High-value workflows
- Install editable with compiled extensions:
  - `python -m pip install -e ".[dev]"`
- Python-only checks before finishing:
  - `python -m pytest tests/ -x`
  - `python -m pytest tests/ --durations=50 --dist=loadscope -n 10 --random-order --verbose`
  - `python -m ruff check aequilibrae/`
- After editing any `.pyx/.pxd/.pxi` under `aequilibrae/paths/cython/`, `aequilibrae/distribution/cython/`, or `aequilibrae/matrix/`, rebuild with `python -m pip install -e ".[dev]"`.
- Docs are Sphinx in `docs/`; from `docs/`, use `make html` (or on Windows, `make.bat html`).

## Repo-specific patterns
- Prefer internal imports `from aequilibrae.log import logger` and `from aequilibrae.parameters import Parameters`; avoid top-level imports inside package modules to prevent circular imports.
- Defaults belong in `parameters.yml` / `Parameters`, not scattered constants.
- Database code uses `?` placeholders and helpers like `commit_and_close`, `safe_connect`, and `connect_spatialite`; do not string-format SQL with user data.
- Schema changes are two-part changes: SQL under `aequilibrae/project/database_specification/...` **and** upgrade handling via `MigrationManager`/project code.
- Pandas 3 compatibility matters here: avoid mutating `.values` in place. Existing code uses `to_numpy(copy=True)` plus assignment back (see `_assign_aggregation_fields` in `aequilibrae/paths/traffic_assignment.py`).
- Logging is file/logger based, not `print`; project loggers are created by `Project.__setup_logger()`.
- Docstrings are reST/Sphinx style with `:Arguments:` / `:Returns:` and runnable examples.

## Integration points and gotchas
- On Windows, SpatiaLite binaries may be auto-downloaded by `ensure_spatialite_binaries()` (`aequilibrae/utils/spatialite_utils.py`). If DB/spatial tests fail, check `AEQ_SPATIALITE_DIR`, PATH, and the copied `proj.db` behavior.
- `setup.py` compiles Cython extensions as C++17 with OpenMP. `AEQ_DEBUG=1` enables debug flags; `AEQ_ASAN=1` enables sanitizer flags where supported.
- Example/integration tests are built from zipped fixtures via `create_example()` and `tests/conftest.py`; common realistic datasets are `sioux_falls`, `nauru`, and `coquimbo`.
- `Network.build_graphs()` filters links by `modes` and converts unsupported links into self-loops before graph prep; if a mode-specific graph looks wrong, inspect the `modes` field in `links`.
- `TrafficClass` construction is strict: matrix index must equal graph centroids, and the matrix computational view must be `float64`.
- Native `.aem` multi-core `computational_view()` requires adjacent cores; OMX loads requested cores into memory.
- Top-level `aequilibrae/__init__.py` is the public API surface; if you change exported names or version, check downstream docs/version references too.

