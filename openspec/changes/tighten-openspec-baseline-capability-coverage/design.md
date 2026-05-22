## Context

AequilibraE now has an OpenSpec baseline for the largest project surfaces, but the baseline is incomplete relative to the scanned implementation. Several public modules, command entry points, result registries, and high-risk modeling workflows have tests and docs but no capability-level requirements.

This change is intentionally a specification-baseline change. It documents existing behavior so future OpenSpec proposals can target the right capability instead of rediscovering the architecture.

## Goals / Non-Goals

**Goals:**

- Add durable current-state requirements for implemented but missing capability areas.
- Keep capabilities small enough that future changes can modify one focused spec.
- Clarify boundaries where existing specs overlap, especially base schema versus migrations and transit import versus transit assignment.
- Avoid changing runtime behavior, public APIs, database schemas, dependencies, or tests.

**Non-Goals:**

- Reorganize source code or test files.
- Split the production transit package into separate import and assignment modules.
- Change assignment, route-choice, distribution, connector, parameter, logging, result, or export behavior.
- Replace current docs with OpenSpec content.

## Decisions

1. Add separate specs for missing public capabilities.

   The new specs mirror implementation ownership rather than compressing everything into existing broad specs. This keeps future changes easier to route: distribution work goes to `distribution-models`, connector work goes to `zoning-and-connectors`, and route-choice/select-link work goes to `route-choice`.

   Alternative considered: fold these behaviors into `graph-and-paths`, `traffic-assignment`, or `network-datamodel`. That would keep fewer files but would blur ownership for large areas that already have distinct modules and tests.

2. Keep `transit-gtfs` as the current transit umbrella, with an explicit boundary requirement.

   Transit import, transit graph construction, preload, and assignment are coupled in the current baseline. The spec now documents that coupling while leaving a clear future path to split assignment into a separate capability when behavior changes require it.

   Alternative considered: create a new `transit-assignment` capability immediately. That would be reasonable later, but this change is a baseline tightening pass and does not need to create a second transit contract unless the implementation changes.

3. Clarify database ownership with added boundary requirements.

   `network-datamodel` owns the current base schema and trigger behavior. `database-migrations` owns evolution of already-created databases. The specs should cross-reference this boundary without duplicating complete schema details.

   Alternative considered: move all schema language into migrations. That would make new-project schema less visible and would not reflect how the code initializes current databases from ordered SQL specifications.

4. Treat results and SimWrapper as one capability for now.

   Project result registration, assignment result persistence, and SimWrapper export all describe user-visible outputs. Keeping them together gives future output-related changes a single first landing zone.

   Alternative considered: separate `simwrapper-export`. That can happen if SimWrapper grows independently from project result persistence.

## Risks / Trade-offs

- Spec drift from over-documenting implementation details: keep requirements behavioral and testable, with code paths named only for traceability.
- Capability overlap remains possible around select-link assignment outputs: route-choice owns route-set/select-link semantics, while traffic assignment owns standard equilibrium assignment workflows.
- Transit remains broad: a future implementation change may need to split `transit-assignment` out of `transit-gtfs`.
- Baseline specs may reveal undocumented edge cases: resolve future conflicts against source code, tests, SQL schema, and docs before changing requirements.

