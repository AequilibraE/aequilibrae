## 1. Validation Report Shape

- [ ] 1.1 Define the demand-connectivity validation return structure for per-class summaries.
- [ ] 1.2 Include class name, matrix core names, total demand, unassignable demand, percentage, OD-pair count, and bounded samples.
- [ ] 1.3 Distinguish missing-centroid demand from present-but-unreachable OD demand.

## 2. Missing Centroid Demand Checks

- [ ] 2.1 Detect centroids that are present in the matrix index but absent from the selected graph.
- [ ] 2.2 Aggregate positive productions and attractions attached to missing graph centroids.
- [ ] 2.3 Suppress demand-loss warnings for missing graph centroids with zero productions and attractions.
- [ ] 2.4 Add tests for mode-disabled connector cases with and without positive demand.

## 3. Reachability Checks

- [ ] 3.1 Reuse existing graph/path connectivity primitives to test reachable destinations for positive-demand origins.
- [ ] 3.2 Aggregate positive OD demand for OD pairs with no path.
- [ ] 3.3 Respect the configured graph mode, time field, centroid-through-flow blocking, and connector availability.
- [ ] 3.4 Add tests for disconnected components, one-way directionality, and connector-only failures.

## 4. Assignment Integration

- [ ] 4.1 Add a public validation helper, such as `TrafficAssignment.validate_demand_connectivity()`.
- [ ] 4.2 Invoke validation automatically before assignment execution.
- [ ] 4.3 Log warnings when any class has positive unassignable demand.
- [ ] 4.4 Keep default behavior warning-only and preserve existing assignment execution behavior.
- [ ] 4.5 Ensure multi-class assignments report class-specific validation summaries.

## 5. Documentation And Verification

- [ ] 5.1 Document the warning behavior in traffic assignment documentation.
- [ ] 5.2 Add examples or notes explaining zero-demand missing centroids versus real demand loss.
- [ ] 5.3 Run focused traffic assignment validation tests.
- [ ] 5.4 Run existing traffic assignment tests.
- [ ] 5.5 Run ruff on touched files.
- [ ] 5.6 Run `openspec.cmd status --change add-assignment-demand-connectivity-warning` and resolve incomplete artifacts.
