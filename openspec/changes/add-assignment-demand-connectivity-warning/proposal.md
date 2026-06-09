## Why

Traffic assignment can silently leave positive OD demand unloaded when centroids are missing from the selected mode graph
or when no path exists between an origin and destination. This is easy to miss in large imported models, especially when
the network and matrix contain zones for multiple modes.

## What Changes

- Add a warning-oriented demand/connectivity validation step for traffic assignment.
- Detect positive demand attached to centroids that are absent from the selected graph, including missing mode-enabled
  connector cases.
- Detect positive OD demand for pairs with no path in the selected graph.
- Report affected demand totals, percentage of class demand, affected OD-pair counts, and small origin/destination
  samples.
- Keep assignment execution warning-only by default; no breaking behavior change is proposed.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `traffic-assignment`: traffic assignment SHALL warn when configured demand cannot be assigned because of missing
  graph centroids, connectors, or unreachable OD pairs.

## Impact

- Affected code: `aequilibrae/paths/traffic_assignment.py`, traffic-class execution setup, graph/path connectivity
  utilities, and assignment tests.
- Affected APIs: likely a public validation helper such as `TrafficAssignment.validate_demand_connectivity()` plus
  automatic invocation before `execute()`.
- No database schema changes are expected.
- No Cython changes are required unless implementation chooses to reuse low-level path results for performance.
