# AGENTS.md

Guidance for Codex, GitHub Copilot, and other coding agents working in this repository.

## Document Roles

- `AGENTS.md` is the operational contract for AI agents: how to work in this repo.
- `openspec/project.md` is the detailed brownfield project context and should be treated as the most complete architecture snapshot.
- `README.md` is the public package overview and quick orientation for users/contributors.

When these documents disagree, check the source files, tests, CI, and SQL schema before editing.

## Project Snapshot

AequilibraE is a Python 3.10+ transportation modeling package. CI currently tests Python 3.10 through 3.14. The package includes traffic assignment, transit and GTFS tooling, trip distribution, matrix handling, geospatial import/export, and SQLite/SpatiaLite-backed project data.

Performance-critical code uses Cython, C++17, and OpenMP. Be conservative around algorithms, numerical behavior, database schemas, and bundled reference datasets.

## Repository Map

- `aequilibrae/` - main Python package
  - `paths/` - graph building, path computation, VDFs, traffic assignment, Cython path algorithms
  - `distribution/` - gravity models and IPF
  - `matrix/` - AequilibraE matrix, OMX, sparse matrix types, Cython matrix code
  - `project/` - project lifecycle, scenarios, network management, database schema, migrations, OSM/GMNS tools
  - `transit/` - GTFS import, route systems, transit graph and assignment support
  - `utils/` - shared utilities and SimWrapper export
  - `reference_files/` - bundled sample/test datasets
- `tests/` - pytest suite and test data
- `docs/` - Sphinx documentation source
- `benchmarking/` - performance benchmarks
- `openspec/` - OpenSpec project context, specs, and changes
- `.codex/skills/` - OpenSpec skills generated for Codex
- `.github/prompts/` and `.github/skills/` - OpenSpec prompts/skills generated for GitHub Copilot
- `.github/copilot-instructions.md` - additional project-specific guidance; consult it when details are missing here
- `pyproject.toml` - project metadata, dependencies, ruff and coverage configuration
- `setup.py` - Cython extension build configuration

## OpenSpec Workflow

OpenSpec is the intended spec-driven layer for this brownfield project. It is initialized with `schema: spec-driven` under `openspec/config.yaml`.

Before substantive code changes:

1. Read this file.
2. Read `openspec/project.md`.
3. Search `openspec/specs/` for capabilities related to the request.
4. If the request changes behavior, public APIs, data model, algorithms, CLI behavior, or documentation promises, create or update an OpenSpec change before implementation.
5. Keep the implementation aligned with the approved proposal, design notes, tasks, and spec deltas.

Small mechanical fixes may proceed without a new OpenSpec change when they do not alter intended behavior. Examples: typos, formatting, narrow lint fixes, obvious test maintenance, or comments. Still mention that no spec change was needed.

When working inside an OpenSpec change:

- Keep `proposal.md` focused on user-visible or maintainer-visible intent.
- Keep `design.md` for technical tradeoffs, migration strategy, algorithmic risk, and compatibility concerns.
- Keep `tasks.md` as an executable checklist and update it as tasks complete.
- Write specs as durable requirements using `SHALL` language and concrete scenarios.
- Do not treat a spec delta as implementation detail; it describes intended behavior.
- When user feedback changes scope, behavior, requirements, data model decisions, API shape, or deferred-work boundaries during an active OpenSpec change, update the affected OpenSpec artifacts before coding. Usually this means `design.md`, `tasks.md`, and any relevant spec delta.
- When a user resolves an open question in conversation, remove or move that question from the open-question list and reflect the decision in the appropriate decision, spec, or task text. Leave only genuinely unresolved or deferred questions visible.
- After implementation, verify tests and docs, then archive or prepare the change according to the OpenSpec workflow.

If OpenSpec commands are available, prefer the native OpenSpec commands. If they are not available, maintain the same folder and document structure manually.

Use `openspec.cmd` from PowerShell if execution policy blocks the npm-generated `openspec.ps1` shim.

## Development Commands

Install in editable development mode:

```bash
pip install -e ".[dev]"
```

CI commonly uses `uv`:

```bash
uv pip install -e ".[dev]"
```

Run the main test suite:

```bash
pytest tests/ --durations=50 --dist=loadscope -n 4 --random-order --verbose
```

Run a quicker local test pass:

```bash
pytest tests/ -x
```

Run linting:

```bash
ruff check aequilibrae/
```

Build docs locally from `docs/`:

```bash
make html
```

On Windows, SpatiaLite setup for tests may require:

```bash
python tests/setup_windows_spatialite.py
```

The wheel workflow currently builds wheels on Ubuntu, Windows, and Ubuntu ARM. The test workflow covers Windows, Ubuntu, and macOS; macOS wheel builds are configured in `pyproject.toml` but skipped by the current wheel workflow/configuration.

## Coding Rules

- Follow existing module style and public API patterns.
- Prefer small, reviewable changes over broad refactors.
- Preserve backward compatibility unless an OpenSpec change explicitly approves a break.
- Use Python 3.10+ syntax.
- Keep ruff's configured 120-character line length.
- Use `logger` from `aequilibrae.log`; avoid `print` in library code.
- Use parameterized SQL queries with `?` placeholders; never format user data into SQL.
- For pandas/NumPy work, avoid unsafe in-place mutation through `.values`; prefer `.to_numpy(copy=...)` and explicit assignment.
- Use existing `Parameters` behavior instead of hardcoding defaults.
- Keep docstrings compatible with the Sphinx/reST style used nearby.
- Treat `Project`, `Graph`, `TrafficAssignment`, `AequilibraeMatrix`, and the database table schemas as public-facing contracts unless an approved spec says otherwise.

## Cython And Performance-Sensitive Code

- Cython files live mainly under `aequilibrae/paths/cython/`, `aequilibrae/distribution/cython/`, and `aequilibrae/matrix/`.
- Rebuild after changing `.pyx`, `.pxd`, or `.pxi` files.
- Be careful with memoryviews, `nogil`, `prange`, and OpenMP behavior.
- Add or update tests around numerical algorithms whenever behavior changes.
- Treat benchmark-impacting changes as design-worthy; document expected performance effects in OpenSpec when applicable.

## Database And Migration Rules

- Database schema, triggers, and migrations live under `aequilibrae/project/database_specification/` and related project modules.
- Schema changes require migration logic and tests.
- Maintain SpatiaLite compatibility.
- Preserve data integrity guarantees enforced by triggers unless an approved spec change says otherwise.
- Remember that project data spans `project_database.sqlite`, `public_transport.sqlite`, `results_database.sqlite`, and matrix files under `matrices/`.

## Security And Robustness Notes

- Do not load pickled graph files from untrusted sources.
- Validate table/field identifiers before dynamic SQL; parameterize all user-provided SQL values.
- Treat migrations as high-risk because they can span multiple databases and must preserve transaction integrity.
- Be explicit about network access in tests or docs examples; OSM and some setup paths can reach external URLs.
- Be careful with SpatiaLite extension loading and `AEQ_SPATIALITE_DIR`, especially on Windows.

## Testing Expectations

- Add focused tests for changed behavior.
- Use existing pytest fixtures and sample data helpers from `conftest.py` and `tests/conftest.py`.
- Integration tests may use bundled datasets such as `nauru.zip` and `sioux_falls.zip`.
- Keep coverage above the configured threshold in `pyproject.toml`.
- If full tests are too expensive, run the narrowest meaningful subset and say what remains unverified.

## Documentation Expectations

- Update Sphinx docs when public APIs, user workflows, model behavior, parameters, or database behavior change.
- Keep examples executable and compatible with doctest expectations where applicable.
- For behavior changes, make sure OpenSpec specs and user-facing docs do not contradict each other.

## Git And Collaboration

- Do not revert user changes or unrelated work.
- Inspect the worktree before large edits.
- Keep changes scoped to the requested task and related spec artifacts.
- Do not commit unless the user asks.
- In reviews, prioritize bugs, behavioral regressions, missing tests, and spec drift.

## Agent Operating Principles

- For brownfield work, understand the existing behavior before changing it.
- Search first with `rg` or `rg --files`.
- Prefer structured parsers and existing APIs over ad hoc text manipulation.
- Ask a concise question only when the next step is blocked by an important ambiguity.
- Report commands run and verification results in the final response.
