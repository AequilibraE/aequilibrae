# Copilot Instructions for AequilibraE

## Project Overview

AequilibraE is an open-source transportation modeling library for Python. It provides:

- **Traffic assignment**: Multi-class user-equilibrium assignment (Frank-Wolfe, Bi-conjugate Frank-Wolfe), all-or-nothing, path-based
- **Transit assignment**: GTFS import with map-matching, optimal-strategies transit assignment
- **Trip distribution**: Gravity model, iterative proportional fitting (IPF)
- **Network management**: SQLite/SpatiaLite-backed project files, OSM import, GMNS support
- **Matrix operations**: OMX support, sparse and compressed matrices
- **QGIS plugin**: QAequilibraE for visualization (flow maps, desire lines, scenario comparison)

Python 3.10+ is required. Performance-critical components are compiled with Cython to C++17 using OpenMP for multi-threading.

## Repository Structure

```
aequilibrae/          # Main package
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
setup.py              # Cython extension compilation
pyproject.toml        # Project metadata and tool configuration
```

## Build Instructions

Install the package in editable mode (compiles Cython extensions):

```bash
pip install -e ".[dev]"
```

For production wheels, `cibuildwheel` is used in CI. On macOS, LLVM (clang) must be used for OpenMP support; GCC is used on Linux; MSVC on Windows.

**Sanitizer builds** (developers only):
- `AEQ_ASAN=1` enables AddressSanitizer (and also UndefinedBehaviorSanitizer on non-Windows platforms)

## Running Tests

```bash
pytest tests/ --durations=50 --dist=loadscope -n 4 --random-order --verbose
```

For a quick single-threaded run:

```bash
pytest tests/ -x
```

Coverage is tracked and must stay above 75%.

## Linting

```bash
ruff check aequilibrae/
```

Config lives in `pyproject.toml` under `[tool.ruff]`. Key settings:
- Line length: 120 characters
- Python target: 3.10+
- Selected rules: B, C, E, F, W (flake8-style)
- Max McCabe complexity: 20

Always run `ruff check` before committing Python changes.

## Code Conventions

### Python

- **Python 3.10+** syntax only; use type hints where they match existing style.
- **Imports**:
  - Use `from aequilibrae.log import logger` (not from `aequilibrae` top-level) to avoid circular imports.
  - Use `from aequilibrae.parameters import Parameters` in internal modules.
- **pandas 3+ compatibility**: Do not mutate `.values` arrays in-place; use `.to_numpy(copy=...)` and assign back.
- **Logging**: Use `logger` from `aequilibrae.log`; avoid bare `print` in library code.
- **Error handling**: Raise `ValueError` for user-input errors; use descriptive messages.
- **Docstrings**: Sphinx/reST-style (e.g., ``:Arguments:``, ``:Returns:``, ``.. code-block::``), matching existing files in the module.

### Cython (`.pyx`, `.pxd`, `.pxi`)

- Cython files live in `aequilibrae/paths/cython/`, `aequilibrae/distribution/cython/`, and `aequilibrae/matrix/`.
- Use `cdef` classes and typed memoryviews for performance-critical code.
- Enable OpenMP via `prange` for parallelism; guard with `nogil` blocks.
- After modifying Cython files, rebuild with `pip install -e ".[dev]"`.

### SQL / SpatiaLite

- Database schema and triggers are in `aequilibrae/project/database_specification/`.
- Use parameterised queries (`?` placeholders) — never string-format SQL with user data.
- Schema changes must be accompanied by migration logic in `aequilibrae/project/`.

## Key Patterns

- **Project object**: `from aequilibrae import Project` — the central entry point for most workflows. Open with `project.open(path)` or create with `project.new(path)`.
- **Graph**: `Graph` objects wrap the network for assignment; load with `project.network.graphs`.
- **Matrix**: `AequilibraeMatrix` wraps OMX files; `AequilibraeData` wraps tabular data.
- **VDF functions**: Implemented as Cython files in `aequilibrae/paths/cython/` (e.g., `bpr.pyx`, `akcelik.pyx`). Note: In the Akcelik VDF, the `tau` parameter already absorbs the factor of 8.
- **Parameters**: `Parameters` class reads/writes `parameters.yml`; use it instead of hardcoding defaults.

## Testing Conventions

- Tests live in `tests/` and mirror the package structure.
- Use `pytest` fixtures; shared fixtures and sample data helpers are in `conftest.py`.
- Parametrise with `@pytest.mark.parametrize` to cover multiple scenarios concisely.
- Integration tests that need a real project use the bundled `nauru.zip` or `sioux_falls.zip` reference datasets from `aequilibrae/reference_files/`.
- Doctest examples in source docstrings are validated by the documentation CI.

## Documentation

Documentation is built with Sphinx from `docs/`. To build locally:

```bash
cd docs && make html
```

- API docs are generated automatically from docstrings.
- Narrative guides and examples live in `docs/source/`.
- The `plot_vdf_functions.py` gallery script regenerates VDF visualization charts when VDF implementations change.
- CI builds both HTML (with gallery) and LaTeX (with `-D plot_gallery=False`).
