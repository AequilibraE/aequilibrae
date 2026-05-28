## 1. Expert Review Setup

- [ ] 1.1 Review `reviewers.md` and confirm the VISUM semantics, AequilibraE integration, and traffic data/counts reviewer roles are sufficient for this change.
- [ ] 1.2 Decide whether reviewer roles will be fulfilled by subagents, humans, or the main implementer with explicit review notes.
- [ ] 1.3 Record the reviewer assignment and review checkpoints in the implementation notes for this change.
- [ ] 1.4 Confirm whether frontend/UI wiring belongs in this repository or an adjacent UI/plugin repository.

## 2. Source Inventory And VISUM Semantics

- [ ] 2.1 Produce a field inventory from sample VISUM layers: `node`, `link`, `connector`, `zone_centroid`, optional `zone_polygon`, `countlocation`, and deferred PT/context layers when present.
- [ ] 2.2 For each inventoried field, record data type, null count, unique categorical values where relevant, sample values, unit pattern, and whether the field is required, optional, directional, derived, preserved-only, or deferred.
- [ ] 2.3 Document semantics for `TSYSSET`, `R_TSYSSET`, empty transport-system values, and asymmetric directional availability.
- [ ] 2.4 Document semantics for `TYPENO`, `R_TYPENO`, `LC`, and `R_LC`, including whether each drives link-type mapping or source metadata preservation.
- [ ] 2.5 Identify all directional `R_*` fields used for private-traffic import and explicitly mark fields that must not be copied across directions.
- [ ] 2.6 Define zone centroid, optional zone polygon, connector, and link-count semantics from the VISUM sample layers.
- [ ] 2.7 Confirm deferred public-transport layers and fields are recognized, reported, and not imported in v1.
- [ ] 2.8 Complete VISUM semantics reviewer checkpoint A for the field inventory and mapping assumptions.

## 3. Mapping Contract And API Shape

- [ ] 3.1 Confirm the public API name and accepted inputs for folder-based conventional-name imports and explicit file-to-layer mapping.
- [ ] 3.2 Define the diagnostics/report object returned or logged by the importer, including warnings, errors, CRS assumptions, deferred layers, mapping choices, and source-record references.
- [ ] 3.3 Define default private-traffic mode mapping with `CAR -> c`, `HGV -> h`, user override behavior for models that merge HGV into car, and mode creation/validation rules for mapped mode IDs.
- [ ] 3.4 Define default link-type mapping from VISUM link classes/types to AequilibraE `link_type` and single-character `link_type_id` values.
- [ ] 3.5 Define deterministic fallback behavior when VISUM classes exceed available or configured link-type IDs.
- [ ] 3.6 Define default assignment-field derivation for capacity, speed, distance, and free-flow time, including unit conversions and selected default capacity period.
- [ ] 3.7 Define source-ID storage per table or mapping structure for imported nodes, links, zones, and connectors, preserving only source fields needed for AequilibraE import, audit, or near-term workflows.
- [ ] 3.8 Define validation severity levels for missing layers, unmapped values, CRS assumptions, topology mismatches, invalid units, and non-assignment-ready records.
- [ ] 3.9 Complete VISUM, AequilibraE, and traffic-data reviewer checkpoint A for the mapping contract and API shape.

## 4. Data Model And Storage Decisions

- [ ] 4.1 Decide whether the importer requires an empty network or supports importing into explicitly targeted non-empty projects.
- [ ] 4.2 Confirm link counts as the first supported traffic-data object and report turn counts, detector/lane counts, screenlines, speeds, routes, and travel-time observations as deferred.
- [ ] 4.3 Decide whether source VISUM length is preserved separately from trigger-derived AequilibraE `distance`.
- [ ] 4.4 Decide whether source-ID preservation uses importer-added core-table columns or a separate source-mapping table.
- [ ] 4.5 Decide where supported link-count associations are stored durably, if any, and keep unsupported count-location fields diagnostic-only.
- [ ] 4.6 Add SQL schema and migration artifacts for link-count or source-mapping storage if required.
- [ ] 4.7 Add tests for any new table list, trigger list, migration listing, metadata documentation, and project upgrade behavior.
- [ ] 4.8 Complete AequilibraE integration reviewer checkpoint B before coding schema-backed storage.

## 5. Fixtures And Test-First Coverage

- [ ] 5.1 Add compact VISUM-like GeoJSON fixtures covering nodes, links, zone centroids, optional zone polygons, connectors, count locations, asymmetric directions, unmapped values, invalid units, missing CRS, and deferred public-transport layers.
- [ ] 5.2 Add tests for field inventory and layer discovery before implementing importer behavior.
- [ ] 5.3 Add tests for default and user-overridden mode mappings, including `CAR` and `HGV` behavior.
- [ ] 5.4 Add tests for default and user-overridden link-type mappings and fallback behavior.
- [ ] 5.5 Add tests for CRS handling with declared, user-supplied, missing, and explicitly accepted default CRS.
- [ ] 5.6 Add tests for unit parsing for length, speed, time, and capacity fields, including unparsable values.
- [ ] 5.7 Add tests for topology validation failures: missing node references, endpoint mismatch, missing zone centroid references, and connector mismatch.
- [ ] 5.8 Add tests for assignment-readiness failures: missing, zero, negative, NaN, or unparsable time and capacity values.
- [ ] 5.9 Add tests that supported link-count associations are preserved even when not usable for assignment validation.
- [ ] 5.10 Add tests that count locations do not modify OD matrices, trigger demand adjustment, or perform calibration.
- [ ] 5.11 Add tests that deferred public-transport layers are reported and not imported into private-traffic network tables.

## 6. Reader And Validation Implementation

- [ ] 6.1 Implement layer discovery for conventional folder file names and explicit file-path handling for VISUM GeoJSON inputs.
- [ ] 6.2 Implement GeoPandas-based layer reading with CRS validation and optional reprojection.
- [ ] 6.3 Implement field inventory generation for diagnostics and reviewer workflows.
- [ ] 6.4 Implement unit parsers for VISUM length, speed, time, and capacity-like fields.
- [ ] 6.5 Implement mapping configuration loading, default mapping application, and user override validation.
- [ ] 6.6 Implement node/link source-reference validation and link geometry endpoint compatibility checks.
- [ ] 6.7 Implement zone centroid, zone polygon, and connector source-reference validation and connector geometry compatibility checks.
- [ ] 6.8 Implement count-location source-reference validation for link-count associations and diagnostics for deferred count-location fields.
- [ ] 6.9 Implement recognized-deferred-layer reporting for stops, stop points, line routes, public-transport-only layers, and OD matrix files.
- [ ] 6.10 Complete reviewer checkpoint C for diagnostics, validation severity, and public API behavior.

## 7. Network Import Implementation

- [ ] 7.1 Implement the public network API entry point and keep behavior consistent with existing `Network.create_from_osm(...)` and `Network.create_from_gmns(...)` patterns.
- [ ] 7.2 Implement mode and link-type creation or validation before any link insert, including default creation/validation of `h` when `HGV` maps to a separate heavy-vehicle mode.
- [ ] 7.3 Import VISUM nodes while preserving source node identifiers and geometry.
- [ ] 7.4 Import VISUM zones and centroids while preserving source zone identifiers and geometry.
- [ ] 7.5 Import VISUM links with AequilibraE direction, modes, link types, geometry, assignment fields, and source metadata.
- [ ] 7.6 Derive assignment-ready numeric fields for distance, speed, capacity, and free-flow time where possible.
- [ ] 7.7 Preserve source VISUM length separately when configured and when it differs from geometry-derived distance.
- [ ] 7.8 Import VISUM connectors as centroid connectors with configured modes, link type, geometry, and source metadata.
- [ ] 7.9 Import or report supported link-count associations without performing demand adjustment.
- [ ] 7.10 Ensure failed imports are transactional or provide explicit partial-import diagnostics and cleanup guidance.

## 8. Graph, Assignment, And Database Verification

- [ ] 8.1 Test trigger behavior after import: `a_node`, `b_node`, `distance`, node `modes`, and node `link_types`.
- [ ] 8.2 Verify imported centroids with `Network.count_centroids()` and graph centroid arrays.
- [ ] 8.3 Verify imported connectors connect centroid nodes to regular network nodes and support configured private modes.
- [ ] 8.4 Run `project.network.build_graphs(fields=[...], modes=["c"])` on imported fixtures.
- [ ] 8.5 Verify `TrafficAssignment.set_time_field(...)` and `set_capacity_field(...)` pass for configured assignment-ready imported fields.
- [ ] 8.6 Verify supported link-count associations can be queried with network associations intact.
- [ ] 8.7 Complete reviewer checkpoint D before documenting the workflow as assignment-ready.

## 9. UI, Documentation, And Examples

- [ ] 9.1 Add or update UI wiring so users can launch the VISUM GeoJSON import workflow from the existing frontend when that frontend lives in this repository.
- [ ] 9.2 Add API docstrings for the VISUM GeoJSON import entry point, mapping configuration, and diagnostics/report object.
- [ ] 9.3 Add user documentation for required layers, optional zone polygons, CRS handling, default mappings, assignment readiness, UI access, link-count scope, and deferred workflows.
- [ ] 9.4 Add a runnable documentation example using compact local fixtures and no external downloads.
- [ ] 9.5 Document count locations as future inputs for validation, calibration, and ODME workflows, not as v1 demand-estimation logic.
- [ ] 9.6 Complete reviewer checkpoint E for deferred public transport, OD matrix import, and count-based OD adjustment.

## 10. Final Verification

- [ ] 10.1 Run focused project/network importer tests.
- [ ] 10.2 Run focused graph and traffic-assignment tests against imported VISUM-like fixtures.
- [ ] 10.3 Run migration/schema tests if count-location or source-mapping storage changes the project database.
- [ ] 10.4 Run relevant documentation doctests or example checks.
- [ ] 10.5 Run `ruff check aequilibrae/` on touched package code.
- [ ] 10.6 Run `openspec.cmd status --change add-visum-geojson-import-pipeline` and resolve any incomplete artifacts.
