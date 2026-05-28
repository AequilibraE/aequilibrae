# Implementation Notes

## Reviewer Assignments

The reviewer charters in `reviewers.md` are sufficient for this change. They cover the three review areas needed before
and during implementation:

- VISUM semantics: source-layer fields, directional values, topology references, mappings, and deferred VISUM scope.
- AequilibraE integration: public API shape, network tables, triggers, zones, centroids, graph building, assignment
  readiness, and schema/migration risk.
- Traffic data and counts: private-traffic assumptions, HGV/CAR handling, assignment fields, link-count associations,
  and explicit deferral of demand adjustment.

Reviewer roles will be fulfilled by the main implementer with explicit review notes because no human or delegated
subagent reviewers have been assigned for this implementation pass. Each checkpoint below must be recorded here before
the corresponding task is marked complete.

## Review Checkpoints

- Checkpoint A, VISUM semantics: complete after the field inventory and mapping assumptions are documented.
- Checkpoint A, mapping contract: complete after API shape, diagnostics, mode mapping, link-type mapping, assignment
  fields, source-ID storage, and validation severity are documented.
- Checkpoint B: complete before adding schema-backed durable source mapping or count-location storage.
- Checkpoint C: complete after reader diagnostics, validation severity, and public API behavior are implemented.
- Checkpoint D: complete after imported VISUM-like fixtures build graphs and pass assignment-readiness checks.
- Checkpoint E: complete after docs and tests explicitly defer public transport, OD matrix import, and count-based demand
  adjustment.

## Frontend/UI Scope

This repository contains the Python package and Sphinx documentation, but not the interactive AequilibraE/QGIS frontend
where file-menu import wiring would normally live. Task 9.1 is therefore scoped to documenting the API and noting that
interactive UI wiring belongs in the adjacent frontend/plugin repository unless such frontend code is later added here.

## Sample Layer Source

No production VISUM GeoJSON files are checked into this repository. Source inventory and tests for this implementation
use compact VISUM-like fixtures that intentionally cover the required fields, directional `R_*` values, CRS cases,
topology references, count-location associations, and deferred public-transport/context layers.

## Checkpoint C

Checkpoint C is approved for the implemented diagnostics and validation behavior. The importer returns structured
diagnostics with severity, code, layer, field, and source-record references; validation errors stop the import before
database writes; non-assignment-ready records are warnings when graph construction can still proceed.

## Checkpoint D

Checkpoint D is approved for the compact VISUM-like fixture coverage. Tests build an AequilibraE graph for mode `c`,
verify centroids/connectors, and exercise `TrafficAssignment.set_time_field(...)` and
`TrafficAssignment.set_capacity_field(...)` against imported assignment fields.

## Checkpoint E

Checkpoint E is approved for the documented v1 scope. Public transport layers, OD matrix import, and count-based demand
adjustment are recognized as deferred and are not imported or executed by this private-traffic network pipeline.

## Verification Notes

- `python -m compileall aequilibrae/project/network/visum_geojson_importer.py aequilibrae/project/network/network.py tests/aeq/project/test_network_visum_geojson.py docs/source/examples/network_manipulation/plot_create_from_visum_geojson.py` passed.
- `ruff check aequilibrae/` passed.
- `uv pip install -e ".[dev]" --system` passed after running through the Visual Studio Build Tools developer command
  environment.
- `python -m pytest tests/aeq/project/test_network_visum_geojson.py -q --basetemp C:/tmp/aequilibrae/.pytest-tmp-visum`
  passed with 13 tests.
- `python docs/source/examples/network_manipulation/plot_create_from_visum_geojson.py` passed.
- No SQL schema, migration, table-list, trigger-list, or durable count/source-mapping table artifacts were added, so the
  migration/schema verification task is not applicable for this implementation slice.
