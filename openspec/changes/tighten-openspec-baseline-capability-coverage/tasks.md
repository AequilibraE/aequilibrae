## 1. Baseline Review

- [ ] 1.1 Review `distribution-models` requirements against distribution source files, tests, and docs.
- [ ] 1.2 Review `zoning-and-connectors` requirements against zoning, connector creation, closest-zone behavior, tests, and docs.
- [ ] 1.3 Review `route-choice` requirements against route choice, path-file, select-link source files, tests, and docs.
- [ ] 1.4 Review `results-and-exports` requirements against result registries, assignment saving, SimWrapper, `aeq-sim`, tests, and docs.
- [ ] 1.5 Review `parameters-and-logging` requirements against parameter loading, project creation/opening/closing, logging tests, and docs.

## 2. Boundary Review

- [ ] 2.1 Confirm `network-datamodel` and `database-migrations` deltas clearly separate base schema ownership from upgrade behavior.
- [ ] 2.2 Confirm `transit-gtfs` deltas clearly state the current transit umbrella boundary and future split condition.
- [ ] 2.3 Check that new specs do not duplicate requirements already owned by `traffic-assignment`, `graph-and-paths`, `matrix-io`, or `project-lifecycle`.

## 3. Validation

- [ ] 3.1 Run `openspec.cmd validate tighten-openspec-baseline-capability-coverage`.
- [ ] 3.2 Run `openspec.cmd validate --all`.
- [ ] 3.3 Update `openspec/project.md` related-documents or baseline-spec list if the change is approved and archived.

