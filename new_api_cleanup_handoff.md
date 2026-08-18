# New API cleanup handoff

## Status and intent

The current branch contains a partially implemented transaction/API rewrite, but it is over-engineered and several
choices are outside the required scope. Treat this document as the authority for the next pass. Prefer direct,
readable code over defensive abstractions. Preserve and adapt existing tests and doctests instead of deleting them.

The cleanup should be delivered as small commits that keep the focused test suite runnable. Do not preserve temporary
APIs from the current branch merely for compatibility.

## 1. Remove the active-project mechanism completely

There is no requirement to support an implicit or process-global active project.

### Required changes

1. Delete the active-project state and accessors from `src/aequilibrae/context.py` and remove their exports.
2. Remove all imports and uses of `get_active_project`, `activate_project`, `Project.activate()`, and
   `Project.deactivate()`.
3. Make dependencies explicit at every call site:
   - algorithms that operate on a project receive a required `Project` argument;
   - scenario-specific code receives a `Scenario` or the narrow gateway/connection manager it needs;
   - `Parameters` receives a path explicitly;
   - closed-project utilities receive a path;
   - migrations receive their connections from the migration runner.
4. Do not retain optional fallbacks such as `project=None` followed by `get_active_project()`.
5. Update generated demo/reference modules, examples, Python, Cython, transit/GTFS code, OSM code, assignment,
   skimming, route choice, gravity, IPF, sub-area, and network simplification.
6. Replace the old active-project tests with tests proving that two explicit `Project` instances do not affect one
   another. Do not delete unrelated coverage.

Useful inventory:

```bash
rg -n 'get_active_project|activate_project|activate_project\(|\.activate\(|\.deactivate\(' src tests docs
rg -n 'project\s*=\s*None|project or ' src/aequilibrae --glob '*.{py,pyx}'
```

Completion means these searches have no application call sites and opening a project has no global side effect.

## 2. Remove project path locking

Project path locking is out of scope. Multiple opens do not need to be detected or prevented by this change.

### Required changes

1. Delete `src/aequilibrae/project/path_lock.py` and remove it from `src/aequilibrae/project/meson.build`.
2. Remove `ProjectPathLock`, `_path_lock`, and all acquire/release logic from `Project.open`, `Project.new`,
   `Project.upgrade`, and `Project.shutdown`.
3. Collapse lock-only wrappers such as `open()`/`_open()`, `new()`/`_new()`, and `upgrade()`/`_upgrade()`.
4. Remove lock-specific tests and lock claims from the migration guides/documentation.
5. Keep ordinary filesystem/database errors descriptive, but do not replace locking with another ownership registry.

## 3. Remove `_require_scenario` by eliminating two-stage initialization

`_require_scenario()` exists because `Project` is allowed to spend part of its lifetime half initialized. Fix that
lifecycle rather than adding guards to every property.

### Target shape

- A usable `Project` is created with a complete root `Scenario` already installed.
- Construction/opening either returns a fully initialized object or raises without publishing one.
- Project properties directly delegate to `self.scenario`; they do not repeatedly call `_require_scenario()`.
- There is no public state in which `root_scenario` or `scenario` is `None` on an otherwise usable project.
- `shutdown()` closes owned resources. Code must not continue using a project after shutdown; closed SQLite resources
  provide the natural failure rather than every property implementing an initialization guard.

### Suggested implementation

1. Move disk validation and complete `Scenario` construction into one factory.
2. Make `Project.from_path(path)` call that factory and pass the finished scenario into `Project.__init__`.
3. Make the new-project factory create required resources and then construct `Project` once. If the existing API name
   `new` must remain, it should return the constructed project rather than mutate an empty shell.
4. Install network, zoning, matrices, results, transit, about, parameters, and logging while constructing the
   `Scenario`, not later in `Project.__load_objects()`.
5. Remove `Scenario.open_candidate`, its `(candidate, created)` protocol, placeholder gateway attributes initialized to
   `None`, `Project.__load_objects`, `_require_scenario`, and the duplicate wrapper methods.
6. Scenario switching may still prepare a complete scenario before swapping it, but it must use the same one-stage
   scenario factory and never expose a partially populated `Scenario`.

This is about object initialization, not path locking or global active-project state.

## 4. Simplify `ConnectionClosure` to the three known databases

A closure is not a generic non-empty string-to-connection mapping. A project database always exists; results and
transit databases are optional. Support no more than these three.

### Target API

```python
closure = ConnectionClosure(
    db_connection=project_connection,
    results_connection=results_connection_or_none,
    transit_connection=transit_connection_or_none,
)

closure.db_connection       # always available
closure.results_connection  # raises RuntimeError when absent
closure.transit_connection  # raises RuntimeError when absent
```

Each property should return that connection's nested transaction manager/wrapper. Remove `__getitem__`, arbitrary
names, mapping iteration, and validation for an arbitrary positive number of connections.

### Behavior

1. `db_connection` is required and corresponds to `project_database.sqlite`.
2. `results_connection` and `transit_connection` are optional.
3. Accessing an absent optional property raises a descriptive `RuntimeError`, for example
   `"This scenario has no results database"`.
4. `transaction()` enters the project manager and only the optional managers that actually exist, using `ExitStack`.
5. `close()` closes only managers that exist.
6. Reject reuse of the same raw SQLite connection in two slots.
7. Update all `closure["project"]`, `closure["results"]`, and `closure["transit"]` call sites to the properties.
8. Add tests for project-only, project+results, project+transit, and all-three closures, including absent-property
   errors and transaction nesting.

Do not add a new generic mapping API to replace `__getitem__`.

## 5. Do not create missing optional databases while opening

Opening/initializing a scenario must reflect what is on disk. It must not create
`results_database.sqlite` or `public_transport.sqlite` as a side effect.

### Required changes

1. Remove `create_auxiliary`, `Path.touch()`, and template copying from `Scenario.open_candidate` or its replacement.
2. Always open `project_database.sqlite`.
3. Open results/transit only when their files already exist; pass `None` to `ConnectionClosure` otherwise.
4. Do not initialize a transit schema merely because opening discovered no transit database.
5. Make result/transit gateway availability explicit:
   - metadata gateways backed by the project DB may still exist;
   - payload/transit operations that need a missing optional DB raise a clear `RuntimeError` when called;
   - do not lazily create the file on first access.
6. Explicit resource-creation workflows may create these files when that is their stated purpose. Ordinary project or
   scenario initialization may not.
7. Add tests that snapshot directory contents before and after open and prove no optional file appears.

## 6. Rewrite `project_table.py` for clarity

`src/aequilibrae/project/project_table.py` is currently too dense. Simplify it before migrating more consumers.

### Remove dynamic record generation

Delete `make_dataclass`, `_record_cache`, `record_name`, and schema-driven `_build_record` type creation. Every table
should define and use a normal, explicit dataclass.

Example:

```python
@dataclass(frozen=True)
class ModeRecord:
    mode_name: str
    mode_id: str
    description: str
    pce: float
    vot: float
    ppv: float

class Modes(ProjectTable):
    record_type = ModeRecord
```

The base table should construct `self.record_type(...)` from an explicit, stable field list. Do not infer a Python type
from `PRAGMA table_info`. User-added fields remain available through `.data`; they do not require runtime dataclass
mutation. Define explicit record dataclasses for links, nodes, modes, link types, periods, zones, matrices, and results.
Keep geometry conversion straightforward and local.

### Allow normal copying

Delete `ProjectTable.__copy__` and `ProjectTable.__deepcopy__`. There is no requirement to prohibit copying. Add a small
test showing `copy.copy()` and `copy.deepcopy()` are not rejected. Do not add replacement copy guards elsewhere.

### General cleanup

- Split validation, SQL construction, row conversion, and bulk operations into short, named helpers.
- Prefer explicit loops and local variables over dense comprehensions where they obscure behavior.
- Keep scalar and DataFrame mutation transaction behavior, but make the control flow obvious.
- Keep `.data` key-index behavior only after auditing and adapting all consumers.
- Restore useful class/method doctests that were removed; mark obsolete examples skipped rather than deleting them.

## 7. Rename and simplify the nested transaction wrapper

`NestedTransactions` is an object, not a collection of transactions. Rename it to something that describes its role,
preferably `NestedTransactionManager`. Rename attributes such as `_transactions` to `_transaction_manager` throughout
gateways and helpers.

### Context-manager contract

Remove the partial DB-API delegation (`execute`, `executemany`, `cursor`, `total_changes`, and the comment describing a
"deliberately limited" API). The transaction context should return the actual persistent `sqlite3.Connection`:

```python
with manager.transaction() as connection:
    connection.execute(...)
```

Use the returned connection in gateway reads and writes. Do not create a second proxy API that mirrors part of
`sqlite3.Connection`. `Project.db_connection` and the three closure properties may expose the manager; callers obtain
SQLite through `manager.transaction()`.

### Keep the implementation close to `nested_transactions_demo.py`

The demo in the repository is the reference implementation. Retain only:

- one raw connection;
- nesting depth;
- a monotonically increasing savepoint identifier;
- outer `BEGIN`/`COMMIT`/`ROLLBACK`;
- nested `SAVEPOINT`/`RELEASE`/`ROLLBACK TO`;
- a fresh context object from each `transaction()` call.

Remove the additional finalization-recovery machinery, exception notes, logging, `_cleanup_failed_finalization`,
`_attach_cleanup_failure`, duplicate-enter state, and other speculative cleanup code. Keep lifecycle handling only where
needed by the closure that owns and closes the connection. Tests should state the simple demo semantics rather than
fault-injecting elaborate recovery behavior.

Because the single-manager context now returns its SQLite connection, update examples and tests that currently assert
that it binds `None`. The project-wide closure context may remain a coordination context without exposing one ambiguous
connection.

## 8. Restore Python-only migrations

The SQL-to-Python migration conversion was intentional. Revert the later change that restored `.sql` migrations and
introduced a statement parser.

### Required restoration

1. Restore the Python migration files:
   - network `000_initial_migration.py` and `001_add_cols_to_results.py`;
   - transit `000_initial_migration.py` and `001_allow_duplicate_nodes.py`;
   - Python mock migrations under `tests/data/mock_migrations`.
2. Point every `migrations.py` inventory back to `.py` files.
3. Delete the restored migration `.sql` files and mock SQL fixtures.
4. Make `Migration.__post_init__` accept Python files only.
5. Delete `iter_sql_statements`, `_comment_only`, `sqlite3.complete_statement()` handling, and all parser tests.
6. Restore the migration comments, docstrings, and tests removed/replaced by commit `996b1d83`. The version immediately
   before that commit (`bd5e7e93`) is a useful source, but adapt restored tests to the simplified manager/closure API
   rather than blindly restoring obsolete raw-connection behavior.
7. Keep migration transaction ownership simple: the runner enters the relevant nested transaction contexts, receives
   their raw SQLite connections, and passes those connections to Python `migrate(...)` functions. Migration functions
   must not find a project globally.
8. Restore the original test cases for initialization, duplicate/negative IDs, status, seen state, ordering, upgrade,
   skip behavior, invalid migration callables, and migration comments. Adapt assertions; do not replace the suite with a
   smaller parser-focused file.
9. Update `docs/source/database_upgrades.rst` to describe Python migrations only. Remove statement termination and
   `complete_statement()` documentation.

There should be no home-grown SQL statement parser and no migration use of `sqlite3.complete_statement()`.

## 9. Restore doctests globally

Doctests and documentation examples were removed or shortened during the API rewrite. Restore them even when an example
cannot yet run.

### Process

1. Compare all touched modules and RST files against the branch before the table rewrite (for example
   `a54e59fc^`) and inventory removed `>>>` blocks and `.. doctest::` directives.
2. Restore those blocks in the replacement gateway/module documentation.
3. Update examples to the final explicit-project API where straightforward.
4. If the underlying feature is temporarily unavailable or intentionally removed, retain the example with
   `# doctest: +SKIP`; do not delete it.
5. Preserve `conftest.py` doctest fixtures and Sphinx's `sphinx.ext.doctest` configuration.
6. Restore tests deleted as part of migration/table work, then adapt or skip them explicitly. The current full project
   test suite has collection failures from removed modules; fix the tests rather than treating a small focused suite as
   sufficient.
7. Run both Python-module and Sphinx doctest jobs used by CI. At minimum verify:

```bash
pytest --doctest-modules src/aequilibrae
sphinx-build -b doctest docs/source docs/_build/doctest
```

Use the repository's actual CI wrapper/options if these commands differ in the build environment.

## 10. Thorough review and cleanup expectations

The next pass should review all code introduced by the transaction rewrite, not only make the mechanical changes above.

### Review checklist

- Remove dead compatibility helpers, duplicate wrappers, stale imports, and comments describing abandoned designs.
- Use consistent names: `connection` for `sqlite3.Connection`, `_transaction_manager` for the nesting wrapper, and
  `connections`/`ConnectionClosure` only for the fixed three-slot owner.
- Do not retain `commit_and_close` or unmanaged connection creation in converted synchronous project code.
- Do not replace simple SQLite behavior with compensation, exception-note, or ownership abstractions unless a current
  test and requirement justify them.
- Keep constructors explicit and fully initialized.
- Do not silently create files or mutate global state during reads/opening.
- Restore existing test coverage before adding new tests; never reduce coverage to make the rewrite pass.
- Run `ruff`, focused transaction/project/table tests, the complete project/transit/path suites, doctests, and an
  end-to-end project smoke test.

Useful final searches:

```bash
rg -n 'get_active_project|activate_project|ProjectPathLock|_path_lock' src tests docs
rg -n '_require_scenario|open_candidate|create_auxiliary' src tests
rg -n 'closure\[|def __getitem__' src/aequilibrae
rg -n 'NestedTransactions|_transactions' src tests docs
rg -n 'make_dataclass|_record_cache|__copy__|__deepcopy__' src/aequilibrae/project
rg -n 'complete_statement|iter_sql_statements|\.sql' src/aequilibrae/project tests/data/mock_migrations
rg -n '>>>|\.\. doctest::' src docs
```

## Suggested commit sequence

1. Remove project locking and its tests/docs.
2. Restore Python-only migrations, their comments, fixtures, and full tests.
3. Simplify and rename the nested transaction manager; update focused tests.
4. Replace generic `ConnectionClosure` with required project plus optional results/transit properties.
5. Stop creating optional database files during open and add file-preservation tests.
6. Make `Scenario` and `Project` one-stage, fully initialized objects; remove `_require_scenario`.
7. Remove active-project state and convert consumers subsystem by subsystem.
8. Simplify `ProjectTable`, add explicit record dataclasses, and migrate table consumers.
9. Restore/adapt all doctests and legacy tests.
10. Run the full review, smoke tests, formatting, and completion searches.

Do not combine all of these into one large commit. Each commit should leave the affected subsystem internally coherent
and should state which tests were run.
