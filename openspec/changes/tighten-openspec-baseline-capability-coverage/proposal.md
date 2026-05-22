## Why

The initial brownfield OpenSpec baseline covers the central project, network, matrix, assignment, transit, migration, and documentation contracts, but several implemented public capabilities still lack durable requirements. Tightening the baseline now reduces drift before OpenSpec becomes the normal gate for future changes.

## What Changes

- Add missing current-state capability specs for distribution models, zoning/connectors, route choice/select-link, results/export behavior, and parameters/logging.
- Clarify that transit GTFS import and transit assignment are both covered today, while documenting boundaries that may justify a future split.
- Clarify ownership between initial database/schema requirements and migration requirements so future schema changes know where to land.
- No runtime code behavior is intended to change in this proposal.
- No breaking changes.

## Capabilities

### New Capabilities

- `distribution-models`: Gravity application/calibration, synthetic gravity model persistence, and iterative proportional fitting behavior.
- `zoning-and-connectors`: Project zoning, centroid nodes, closest-zone lookup, and centroid connector creation.
- `route-choice`: Route choice set generation, path-file output, link loading, and select-link behavior.
- `results-and-exports`: Project result registry, assignment/result persistence, SimWrapper export, and `aeq-sim` command behavior.
- `parameters-and-logging`: YAML parameter loading, active-project parameter precedence, package defaults, and project logging behavior.

### Modified Capabilities

- `network-datamodel`: Clarify that it owns base table/schema and trigger contracts, while schema evolution belongs to database migrations.
- `database-migrations`: Clarify that it owns schema evolution, migration state, ordering, and upgrade behavior rather than the complete base schema contract.
- `transit-gtfs`: Clarify current boundaries between GTFS/transit graph behavior and transit assignment behavior.

## Impact

- Affected OpenSpec artifacts: new spec files under `openspec/specs/` when archived, plus deltas for `network-datamodel`, `database-migrations`, and `transit-gtfs`.
- Affected implementation areas for future traceability: `aequilibrae/distribution/`, `aequilibrae/project/zoning.py`, `aequilibrae/project/network/connector_creation.py`, `aequilibrae/paths/route_choice*.py`, `aequilibrae/paths/cython/route_choice_*`, `aequilibrae/project/results.py`, `aequilibrae/paths/results/`, `aequilibrae/utils/simwrapper/`, `aequilibrae/parameters.py`, and `aequilibrae/log.py`.
- Affected tests for future verification: distribution, connector creation, select-link/route-choice, results, SimWrapper, parameters, logging, transit graph, and migration tests.
- This is a specification-baseline change only; implementation and public API behavior should remain unchanged.
