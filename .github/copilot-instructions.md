# Instructions for AequilibraE

## Project Overview

AequilibraE is an open-source transportation modeling library for Python. It provides:

- **Traffic assignment**: Multi-class user-equilibrium assignment (Frank-Wolfe, Bi-conjugate Frank-Wolfe), all-or-nothing, path-based
- **Transit assignment**: GTFS import with map-matching, optimal-strategies transit assignment
- **Trip distribution**: Gravity model, iterative proportional fitting (IPF)
- **Network management**: SQLite/SpatiaLite-backed project files, OSM import, GMNS support
- **Matrix operations**: OMX support, sparse and compressed matrices
- **QGIS plugin**: QAequilibraE for visualization (flow maps, desire lines, scenario comparison)

Python 3.11+ is required. Performance-critical components are compiled with Cython to C++17 using OpenMP for multi-threading.

## Repository Structure

```
src/aequilibrae/      # Main package
  paths/              # Traffic assignment, graph, VDF, Cython path algorithms
  distribution/       # Trip distribution (gravity, IPF)
  matrix/             # AequilibraE matrix, OMX, sparse matrix types
  project/            # Project/network management, OSM/GMNS builders, database
  transit/            # Transit assignment and GTFS tools
  utils/              # Shared utilities, SimWrapper export
  reference_files/    # Bundled test/sample datasets (spatialite.sqlite, etc.)
  parameters.yml      # Default model parameters
docs/                 # Sphinx documentation source
tests/                # pytest test suite
benchmarking/         # Performance benchmarks
meson.build           # Meson/Cython extension compilation
.spin/cmds.py         # Shared lint and doctest commands
pyproject.toml        # Project metadata, dependency groups and Spin/tool configuration
```

## Build Instructions

Run development commands from the repository root in an activated virtual environment. Install dependencies without installing AequilibraE, then let Spin build the package:

```bash
uv pip install -r pyproject.toml --extra create --group dev
spin build
```

`-r pyproject.toml` installs runtime dependencies; `--extra create` adds network-creation dependencies, and `--group dev` includes the build tools, Spin, tests and documentation dependencies. Spin builds with Meson/Ninja into `build/` and stages the package in `build-install/`.

`spin test`, `spin doctest`, `spin docs` and the Spin environment commands build automatically and use the staged package. Use `spin python`, `spin ipython` or `spin run COMMAND` rather than plain Python when working against that build. `--no-build` skips compilation, but still expects an existing Spin build; it does not select an installed wheel. If an editable installation of this checkout exists, Spin skips its separate build and relies on editable import-time rebuilds.

For production wheels, `cibuildwheel` is used in CI. On macOS, LLVM (clang) must be used for OpenMP support; GCC is used on Linux; MSVC on Windows.

**Sanitizer builds** (developers only):

```bash
spin build -C build-sanitised -- -Dbuildtype=debug -Db_sanitize=address,undefined
```

Use `-C build-sanitised` for subsequent Spin commands. Running sanitised tests on Linux also requires the runtime preload and suppression settings shown in `.github/workflows/unit_tests.yml`; `AEQ_ASAN` is not a supported build switch.

## Running Tests

```bash
spin test
spin doctest
```

`spin test` defaults to eight pytest workers, global random ordering and reporting the 50 slowest tests. `spin doctest` runs the shared Python/RST doctest targets serially. Both build the package first unless an editable installation is detected.

For a quick single-worker run or a targeted check:

```bash
spin test -j 1 -- tests/ -x
spin test -j 2 -- tests/aequilibrae/paths -k graph
```

Arguments after `--` go to pytest and replace the configured default pytest argument list, rather than extending it. Use `-j` to control worker count.

```bash
spin test --coverage
```

Coverage must be at least 75%; Spin writes the HTML report to `build/coverage/`. The main unit-test CI matrix deliberately uses a non-editable UV installation followed by direct pytest to validate the installed wheel; do not replace those invocations with `spin test`. The coverage CI job uses Spin to test its Meson build.

## Linting

```bash
spin lint
```

This runs `ruff check`, `ruff format --check`, cython-lint, clang-tidy and clang-format without modifying files. Install clang-tidy and clang-format with your system package manager. Linting does not build or configure the package: clang-tidy needs `build/compile_commands.json` from a Meson setup/build. If it is missing, only clang-tidy is skipped with a warning. Use `spin lint -C BUILD_DIR` for another build directory.

Config lives in `pyproject.toml` under `[tool.ruff]`. Key settings:
- Line length: 120 characters
- Python target: 3.11
- Selected rules: B, C, E, F, W (flake8-style), I (import ordering), RUF022 (sorted `__all__`) and TID252 (absolute library imports; tests may use relative imports)
- Max McCabe complexity: 20

Always run `spin lint` before committing Python or native-code changes.

## Code Conventions

### Python

- **Python compatibility**: All code must run on Python 3.11; do not use syntax or standard-library APIs requiring a newer version without a compatible fallback.
- **Type hints**: Annotate new or substantially changed public APIs. Prefer built-in generics (`list[str]`) and unions (`T | None`). Keep annotation changes scoped to the task.
- **Imports**: Use `from aequilibrae.parameters import Parameters` in internal modules. Avoid importing internal dependencies through the package's top-level re-exports, which can cause circular imports.
- **pandas 3+ compatibility**: Prefer direct pandas assignment, such as `.loc[...] = ...`. Do not mutate `.values` arrays in-place. If an algorithm requires a writable NumPy buffer, use `.to_numpy(copy=True)`, mutate it, then explicitly assign the result back. `.to_numpy(copy=False)` may return a read-only array; do not force shared arrays writable.
- **Logging**: Use `import logging` and a module-level `logger = logging.getLogger(__name__)`. Avoid bare `print` in library code.
- **Error handling**: Use `TypeError` for inappropriate argument types and `ValueError` for invalid values of an appropriate type. Preserve specific operational exceptions such as `FileNotFoundError` and `PermissionError`, and existing public exception contracts. Use descriptive messages and explicit chaining (`raise ... from exc`) when translating exceptions.
- **Docstrings**: Use standard Sphinx/reST fields (`:param name:`, `:returns:`, `:raises ValueError:`) for new or substantially revised API documentation. Describe current behavior and contracts, not historical behavior or comparisons with previous implementations. Keep examples in reST directives such as `.. code-block:: python`. Keep documentation changes scoped to the task.

### Cython (`.pyx`, `.pxd`, `.pxi`)

- Cython files live in `src/aequilibrae/routing/cython/`, `src/aequilibrae/distribution/cython/`, and `src/aequilibrae/matrix/`.
- Use `cdef` classes and typed memoryviews for performance-critical code.
- Use OpenMP via `prange` in `nogil` regions only when iterations are independent or use valid reductions. Releasing the GIL does not make shared writes safe: make ownership of mutable buffers explicit. Benchmark changes, use workload thresholds where appropriate, and account for oversubscription from pytest workers and other thread pools.
- After modifying Cython files, rebuild with `spin build`, or let `spin test` rebuild before testing. Editable installations rebuild on import instead.

### SQL / SpatiaLite

- Database schema and triggers are in `src/aequilibrae/project/database_specification/`.
- Use parameterised queries (`?` placeholders) — never string-format SQL with user data.
- Schema changes must be accompanied by migration logic in `src/aequilibrae/project/`.

## Key Patterns

- **Project object**: `from aequilibrae import Project` — the central entry point for most workflows. Open with `Project.open(path)` or create with `Project.create(path)`; both return ready-to-use context managers.
- **Network**: `project.tables` manages database tables; `project.tables.build()` copies them into an independent `aequilibrae.routing.Network`. Construct `Network(links, ...)` for in-memory data. Network configuration fields are fixed; use `dataclasses.replace()` for validated replacements, and construct a new Network after topology or row-order changes.
- **Graph**: Build fixed-topology routing graphs with `network.build_graph(mode, cost, ...)` or the project network's `build_graph()` convenience method. Graphs expose `routing`, `mapping` and `skimming`, not source DataFrames. Change directed network-order costs with `graph.set_costs()`; rebuild for other configuration changes. Link contraction (`compact`) and dead-end removal (`remove_dead_ends`) are independent, opt-in options requiring centroids.
- **Assignment**: Configure `TrafficClass` independently of graphs; routing policies belong in `graph_options`. `TrafficAssignment` builds each class's graph internally and enables contraction and dead-end removal by default. Pass `project=project` when using an independent network and saving project outputs.
- **Path results**: `PathResults.path` contains local graph-link indices. Use `get_network_links()` for physical link IDs and directions. Both A* heuristics require explicit coordinates and a heuristic scale; graphs do not retain coordinates.
- **Matrix**: `AequilibraeMatrix` holds dense matrices in memory; `AequilibraeSparseMatrix` holds SciPy sparse arrays in memory. Both share a zone index across their matrices. `MatrixStore` provides OMX-backed dense and sparse storage; prefer a `with` block to manage its lifetime. Store reads return independent arrays, so assign changes back explicitly.
- **VDF functions**: Implemented as Cython files in `src/aequilibrae/routing/cython/` (e.g., `bpr.pyx`, `akcelik.pyx`). Note: In the Akcelik VDF, the `tau` parameter already absorbs the factor of 8.
- **Parameters**: `Parameters` class reads/writes `parameters.yml`; use it instead of hardcoding defaults.

## Testing Conventions

- Tests live in `tests/` and mirror the package structure.
- Use `pytest` fixtures; shared fixtures and sample data helpers are in `conftest.py`.
- Parametrise with `@pytest.mark.parametrize` to cover multiple scenarios concisely.
- Integration tests that need a real project use the bundled `nauru.zip` or `sioux_falls.zip` reference datasets from `src/aequilibrae/reference_files/`.
- Doctest examples in source docstrings are validated by the documentation CI.

## Documentation

Documentation is built with Sphinx from `docs/`. To build locally:

```bash
spin doctest
spin docs
```

`spin docs` builds HTML against the local build (or editable installation). Use `spin docs --no-plot` to skip gallery execution and `spin docs targets` to list available targets.

- API docs are generated automatically from docstrings.
- Narrative guides and examples live in `docs/source/`.
- `spin doctest` only collects selected docstrings and RST paths; it does not execute gallery scripts. Validate changed examples with the local build or a full `spin docs` run. `spin docs --no-plot` skips gallery execution, and Sphinx-Gallery may reuse cached results for unchanged scripts.
- The `plot_vdf_functions.py` gallery script regenerates VDF visualization charts when VDF implementations change.
- CI builds both HTML (with gallery) and LaTeX (with `-D plot_gallery=False`).
