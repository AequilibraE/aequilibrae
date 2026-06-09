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

## Post-Review Connector Compatibility

Real VISUM GeoJSON connector exports may omit a numeric connector `NO` property and instead expose topology through
`ZONENO`, `NODENO`, `DIRECTION`, and `R_DIRECTION`, with a vendor-specific feature ID. The importer now treats connector
`NO` as optional, keeps `visum_connector_no` only when numeric source values are present, and stores a deterministic
`visum_connector_key` derived from zone, node, and usable mapped direction. Duplicate generated keys receive stable
numeric suffixes.

Post-review verification:

- `python -m pytest tests/aeq/project/test_network_visum_geojson.py -q --basetemp C:/tmp/aequilibrae/.pytest-tmp-visum`
  passed with 15 tests and 1 skipped opt-in external-data test.
- `ruff check aequilibrae/project/network/visum_geojson_importer.py tests/aeq/project/test_network_visum_geojson.py tests/conftest.py`
  passed.
- `python -m compileall aequilibrae/project/network/visum_geojson_importer.py tests/aeq/project/test_network_visum_geojson.py`
  passed.
- The opt-in external-data smoke test for `C:\Users\Pablo Barceló\Downloads` progressed past connector source-ID
  handling and now fails on unmapped transport systems such as `BUS`, `TRAIN`, and `TRAM`, which is a separate
  mode-filtering/mapping decision.

## Post-Review Transport-System Mapping Policy

VISUM transport systems outside the default `CAR -> c` and `HGV -> h` mapping now require an explicit caller decision:
they must be included in `mode_mapping` or listed in `ignored_transport_systems`. Explicitly ignored transport systems
are reported with diagnostics. Records whose declared transport systems are all ignored, or whose transport systems are
empty in both directions, are skipped and reported rather than imported with an empty AequilibraE `modes` value.

Post-review verification:

- `python -m pytest tests/aeq/project/test_network_visum_geojson.py -q --basetemp C:/tmp/aequilibrae/.pytest-tmp-visum`
  passed with 20 tests and 1 skipped opt-in external-data test.
- `ruff check aequilibrae/project/network/visum_geojson_importer.py aequilibrae/project/network/network.py tests/aeq/project/test_network_visum_geojson.py tests/conftest.py`
  passed.
- A real-folder import using `ignored_transport_systems={'BUS', 'TRAIN', 'TRAM', 'BIKE', 'WALK', 'PUTW'}` progressed
  past transport-system validation and then failed on a separate link-type uniqueness collision in the real VISUM link
  classes.

## Post-Review Numeric Link-Type Naming

Real VISUM exports can include numeric `TYPENO` fallback values when `LC` is missing. AequilibraE link-type names only
allow letters and underscores, so numeric source values are now encoded as words in generated names. For example,
`TYPENO=2` becomes `visum_two` and `TYPENO=92` becomes `visum_nine_two`, avoiding collisions while satisfying the
existing link-type API contract.

Post-review verification:

- `python -m pytest tests/aeq/project/test_network_visum_geojson.py -q --basetemp C:/tmp/aequilibrae/.pytest-tmp-visum`
  passed with 21 tests and 1 skipped opt-in external-data test.
- `ruff check aequilibrae/project/network/visum_geojson_importer.py aequilibrae/project/network/network.py tests/aeq/project/test_network_visum_geojson.py tests/conftest.py`
  passed.
- A real-folder import using `ignored_transport_systems={'BUS', 'TRAIN', 'TRAM', 'BIKE', 'WALK', 'PUTW'}` progressed
  past link-type creation and now fails on a separate node integrity trigger: duplicate/on-top node geometries in the
  source `node.geojson`.

## Post-Review Coincident Node Compatibility

The Karlsruhe VISUM export includes coincident nodes that encode separate modal/topological layers rather than duplicate
data. AequilibraE rejects exact duplicate node coordinates, and merging these source nodes could create artificial
connectivity while turn restrictions remain outside scope. The importer now preserves distinct source node IDs by
default with `duplicate_node_policy="offset"`, stores original VISUM coordinates, rewrites imported link/connector
endpoints to the adjusted coordinates, and reports duplicate coordinate groups. Callers can use
`duplicate_node_policy="error"` to retain strict rejection behavior.

The same Karlsruhe smoke check then exposed regular node IDs that collide with zone centroid IDs. The importer now keeps
zone IDs as centroid node IDs, remaps only the conflicting regular node IDs, preserves `visum_node_no`, reports
`node-id-remapped`, and inserts links/connectors through the source-to-imported node mapping.

The following Karlsruhe smoke check then exposed zone centroid coordinates that collide with regular network node
coordinates. The importer now offsets such centroid node geometries, preserves original VISUM centroid coordinates, and
rewrites connector starts to the adjusted centroid coordinates.

## Compact Imported Link IDs

VISUM source link numbers in the Karlsruhe export are sparse and high-valued, with source IDs above two billion. The
graph compression path allocates by maximum `link_id`, so the importer now assigns compact AequilibraE `links.link_id`
values and preserves source IDs in `links.visum_link_no`, `links.visum_connector_no`, `links.visum_connector_key`, and
`report.source_references`.

The Karlsruhe project was regenerated from the local Downloads VISUM folder with explicit mappings
`CAR -> c`, `HGV -> h`, `BIKE -> b`, `WALK -> w`, and `BUS`/`PUTW`/`TRAIN`/`TRAM -> t`, then
`Visum_3_modes.omx` was imported as `visum_3_modes`.

Verification:

- `python -m pytest tests/aeq/project/test_network_visum_geojson.py -q --basetemp C:/tmp/aequilibrae/.pytest-tmp-visum`
  passed with 26 tests and one opt-in external test skipped.
- `python -m pytest tests/aeq/project/test_matrices.py -q` passed with 13 tests.
- `ruff check aequilibrae/project/network/visum_geojson_importer.py tests/aeq/project/test_network_visum_geojson.py aequilibrae/project/data/matrices.py tests/aeq/project/test_matrices.py` passed.
- Regenerated Karlsruhe `links.link_id` values are compact from 1 through 13,723 while `visum_link_no` still preserves
  source IDs up to 2,030,064,882.
- `Visum_3_modes.omx` contains cores `Car`, `HVG`, and `PUT`; its `NO` mapping matches all 726 zones and centroid nodes.

## Connector Assignment Defaults

The Karlsruhe GeoJSON and Shapefile connector exports omit connector travel-time fields. A VISUM SQLite export inspected
later showed richer connector columns such as `T0_TSYS(CAR)`, so a future SQLite importer should prefer those source
values when available. For GeoJSON, the importer now defaults connector assignment values only when a connector has
usable imported modes but lacks positive assignment fields: connector travel time is derived from connector length using
a deterministic 30 km/h fallback speed, missing/zero length can be derived from connector geometry, and capacity defaults
to 99,999. Each default emits structured diagnostics.

The graph builder also now prevents links excluded from a requested mode from poisoning that mode's compressed graph with
missing numeric fields while those excluded rows are being converted to self-loops.

Post-default verification:

- `python -m pytest tests/aeq/project/test_network_visum_geojson.py::test_connector_assignment_fields_default_when_not_exported tests/aeq/project/test_network_visum_geojson.py::test_mode_excluded_missing_fields_do_not_poison_car_graph -q --basetemp C:/tmp/aequilibrae/.pytest-tmp-visum-new` passed.
- `ruff check aequilibrae/project/network/visum_geojson_importer.py aequilibrae/project/network/network.py tests/aeq/project/test_network_visum_geojson.py` passed.
- Regenerated Karlsruhe from `C:\Users\Pablo Barceló\Downloads\Karlsruhe`, imported 8,432 nodes, 726 zones, 10,902 links,
  2,821 connectors, and imported `Visum_3_modes.omx` as `visum_3_modes`.
- The regenerated car graph has 8,367 compressed nodes, 726 zones, 25,041 directed graph rows, zero NaN `travel_time`
  values, zero NaN `capacity` values, minimum travel time 0.0013561301794359395, and minimum capacity 1.0.
- A 5-iteration MSA traffic assignment using the `Car` matrix core completed and returned 13,723 result rows with
  `Car_tot` sum 35,805,710.82760005.
- The car graph still warns that 28 centroids (`2000115` through `2000142`) are not present in the compressed graph.
  Assignment completes, but those zones should be reviewed against VISUM connectivity if exact demand coverage is needed.
