# AequilibraE 2.0 library cutover playbook

This is the implementation ledger for the persistent-connection table API. A
wave is complete only when every search hit is converted or recorded as an
explicitly path-owned, closed-project utility.

## Final dependency and ownership graph

- `Project` owns a root `Scenario` for its whole open lifetime and references
  one selected scenario. It owns the canonical-path OS lock.
- Each `Scenario` owns paths, logging identity/handler, parameters, all
  gateways, and one `ConnectionClosure` named `project`, `results`, `transit`.
- `ConnectionClosure[name]` is the sole `NestedTransactions` owner for one
  distinct SQLite connection. Project/spatial gateways share `project`.
- A table gateway receives only its named manager. Coordinators may receive an
  immediate domain owner, paths, or sibling gateway, never `Project` as a
  connection service locator.
- Standalone mutations enter `manager.transaction()`. A project transaction
  enters all managers with `ExitStack`; nested scopes are savepoints. Contexts
  bind `None`. Multi-file commit is coordinated, not distributed-atomic.
- Generic matrix/result CRUD is metadata-only. `create`, `delete_matrix`, and
  `delete_result` own and compensate external resources, and reject an active
  enclosing database transaction.

## Ordered waves

1. **Connection primitives:** land `NestedTransactions` and
   `ConnectionClosure`, foreign keys, failure cleanup, and tests. No raw
   connection may leave a factory without immediately acquiring an owner.
2. **Migrations:** pass closures, establish one transaction owner, replace
   `executescript` with complete-statement iteration, and terminate all SQL.
3. **Scenario lifecycle:** stage all three files, create the closure and all
   gateways off-object, then install. Add canonical path locking, static closed
   `Project.upgrade(path)`, deterministic shutdown, and two-phase switching.
4. **Gateway cutover:** inject managers; remove connection arguments and
   `TableBatch`; make `.data` key-indexed; migrate every consumer.
5. **Resource helpers:** first finalize compensated matrix/result signatures,
   then convert every Python and Cython producer in one pass.
6. **Global-state removal and cleanup:** delete active-project/context and
   connection aliases, update generated files/docs/examples, run all ledgers.

Do not begin producer conversion before the public resource signatures and user
migration guide are reviewed. Do not delete global fallback code before its
callers have explicit dependencies.

## Search inventory and ledger

Run from the repository root and paste every hit into the subsystem ledger in
the implementing PR. Search Python, Cython, RST/Markdown, examples, tests,
templates, and generated reference files.

```bash
rg -n 'get_active_project|activate_project|activate\(|deactivate\('
rg -n 'database_connection|database_path\('
rg -n 'db_connection_spatial|results_connection|transit_connection'
rg -n 'commit_and_close|manual_transaction|\.commit\(|\.rollback\(|\.close\('
rg -n 'with .*connection|executescript|BEGIN |COMMIT|ROLLBACK|SAVEPOINT'
rg -n 'conn\s*=|def .*conn' src tests docs
rg -n 'TableBatch|\.batch\('
rg -n 'new_record|set_data|\.save\('
rg -n 'DataFrame\.to_sql|\.to_sql\('
rg -n '\.(links|nodes|modes|link_types|periods|zoning)\.data|\.data'
rg -n '\.(link_id|node_id|mode_id|link_type|period_id|zone_id)\b'
```

Ledger columns: `path:line`, subsystem, old ownership assumption, conversion,
test/smoke command, status. The only justified connection exception is a
closed-project operation that creates and destroys its own closure in one
function; record its owner and teardown boundary.

## Call-site recipes

### Connections and transactions

- Routine open-project SQL: pass `scenario.connections[name]`; call delegated
  `execute`/`executemany`; enter `.transaction()` only if this operation owns a
  mutation scope.
- Gateway: constructor accepts one manager. Reads execute directly; every
  mutation enters that manager's transaction, even for one statement.
- Cross-gateway domain operation: inject the domain owner/sibling gateways and
  wrap each multi-statement named-database operation in a savepoint.
- Migration: receive `ConnectionClosure`; only the manager/initializer enters
  `closure.transaction()`. Helpers never finalize or nest.
- Worker/one-off closed file utility: create raw connections and immediately
  transfer all to an operation-owned closure; close it in `finally`.
- Never substitute a native connection context, `executescript`, explicit
  transaction SQL, or a second connection to the same scenario database.

### Explicit dependencies

Pass `Project` only to a public algorithm whose contract genuinely spans
scenario domains. Pass `Scenario` when selection-specific paths and several
named managers are required. Pass a manager to a gateway/SQL helper, a path to
a filesystem-only helper, and a gateway/domain object where its behavior is
required. `Parameters` always receives a path.

### Key-indexed frames

- Identity/filter/allocation: use `frame.index`, `index.to_numpy()`, `.loc`, and
  index-aware joins.
- Remove redundant `.set_index(key)` and key-column selections.
- At exports, query engines, column merges, or external APIs that require an ID
  column, use a local `frame.reset_index()` and document the boundary.
- `update_from`: preserve the named key index; do not reset it or copy key into
  columns.
- `insert_from`: supply key as a value column when explicit keys are required.

Audit sub-area, connector creation, zoning, network simplifier, GMNS export,
GTFS, and transit graph builder explicitly; passing current tests is not enough.

### Matrix/result producers

Replace mutable `new_record`/assignment/`set_data`/`save` sequences with one
resource helper call carrying payload and complete metadata, including reports.
Use a registration helper only when the caller intentionally owns an existing
resource. Never pre-create a final path and then insert metadata. Results must
persist all index levels, so preserve `link_id`/other identity as the DataFrame
index before calling `Results.create`.

Inventory gravity, IPF, network skimming, assignment matrices/skims/results and
select-link output, route choice, optimal-strategy/public-transport Cython,
Delaunay, examples, and generated demo code.

## Subsystem verification

```bash
pytest -q tests/aequilibrae/utils -k 'transaction or connection'
pytest -q tests/aequilibrae/project/test_migration_manager.py
pytest -q tests/aequilibrae/project -k 'project_table or transaction or scenario'
pytest -q tests/aequilibrae/project/data -k 'matrices or results'
pytest -q tests/aequilibrae/transit tests/aequilibrae/paths -k 'save or graph or gtfs'
ruff check <all changed Python files>
```

Run a Sioux Falls smoke: mixed link/node/mode rollback then commit; a partially
valid DataFrame update rollback; metadata-only and resource-aware matrix/result
deletions; shutdown followed by rejected manager use. Run save-to-project smoke
coverage for every producer listed above and compile/test changed `.pyx` files.

## Completion searches

All removed-symbol searches above must return no application/docs/examples hits.
Additional checks:

```bash
rg -n 'TableBatch|\.batch\(|database_connection|db_connection_spatial|results_connection|transit_connection'
rg -n 'get_active_project|activate_project|commit_and_close|manual_transaction'
rg -n 'new_record|set_data|DataFrame\.to_sql|\.to_sql\(' src docs examples tests
rg -n 'def [^(]+\([^)]*conn\s*=|def [^(]+\([^)]*conn[,:)]' src/aequilibrae/project
```

Confirm manager public surfaces have no `commit`, `rollback`, `close`, or raw
connection; every connection has foreign keys enabled; all scenarios always own
three managers; no migration helper finalizes; every SQL file parses; every
`.data` key assumption has an explicit index/reset decision; logs are identity
filtered; path locks and staging clean up on every injected failure.
