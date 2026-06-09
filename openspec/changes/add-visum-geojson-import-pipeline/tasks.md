## 1. Expert Review Setup

- [x] 1.1 Review `reviewers.md` and confirm the VISUM semantics, AequilibraE integration, and traffic data/counts reviewer roles are sufficient for this change.
- [x] 1.2 Decide whether reviewer roles will be fulfilled by subagents, humans, or the main implementer with explicit review notes.
- [x] 1.3 Record the reviewer assignment and review checkpoints in the implementation notes for this change.
- [x] 1.4 Confirm whether frontend/UI wiring belongs in this repository or an adjacent UI/plugin repository.

## 2. Source Inventory And VISUM Semantics

- [x] 2.1 Produce a field inventory from sample VISUM layers: `node`, `link`, `connector`, `zone_centroid`, optional `zone_polygon`, `countlocation`, and deferred PT/context layers when present.
- [x] 2.2 For each inventoried field, record data type, null count, unique categorical values where relevant, sample values, unit pattern, and whether the field is required, optional, directional, derived, preserved-only, or deferred.
- [x] 2.3 Document semantics for `TSYSSET`, `R_TSYSSET`, empty transport-system values, and asymmetric directional availability.
- [x] 2.4 Document semantics for `TYPENO`, `R_TYPENO`, `LC`, and `R_LC`, including whether each drives link-type mapping or source metadata preservation.
- [x] 2.5 Identify all directional `R_*` fields used for private-traffic import and explicitly mark fields that must not be copied across directions.
- [x] 2.6 Define zone centroid, optional zone polygon, connector, and link-count semantics from the VISUM sample layers.
- [x] 2.7 Confirm deferred public-transport layers and fields are recognized, reported, and not imported in v1.
- [x] 2.8 Complete VISUM semantics reviewer checkpoint A for the field inventory and mapping assumptions.

## 3. Mapping Contract And API Shape

- [x] 3.1 Confirm the public API name and accepted inputs for folder-based conventional-name imports and explicit file-to-layer mapping.
- [x] 3.2 Define the diagnostics/report object returned or logged by the importer, including warnings, errors, CRS assumptions, deferred layers, mapping choices, and source-record references.
- [x] 3.3 Define default private-traffic mode mapping with `CAR -> c`, `HGV -> h`, user override behavior for models that merge HGV into car, and mode creation/validation rules for mapped mode IDs.
- [x] 3.4 Define default link-type mapping from VISUM link classes/types to AequilibraE `link_type` and single-character `link_type_id` values.
- [x] 3.5 Define deterministic fallback behavior when VISUM classes exceed available or configured link-type IDs.
- [x] 3.6 Define default assignment-field derivation for capacity, speed, distance, and free-flow time, including unit conversions and selected default capacity period.
- [x] 3.7 Define source-ID storage per table or mapping structure for imported nodes, links, zones, and connectors, preserving only source fields needed for AequilibraE import, audit, or near-term workflows.
- [x] 3.8 Define validation severity levels for missing layers, unmapped values, CRS assumptions, topology mismatches, invalid units, and non-assignment-ready records.
- [x] 3.9 Complete VISUM, AequilibraE, and traffic-data reviewer checkpoint A for the mapping contract and API shape.

## 4. Data Model And Storage Decisions

- [x] 4.1 Decide whether the importer requires an empty network or supports importing into explicitly targeted non-empty projects.
- [x] 4.2 Confirm link counts as the first supported traffic-data object and report turn counts, detector/lane counts, screenlines, speeds, routes, and travel-time observations as deferred.
- [x] 4.3 Decide whether source VISUM length is preserved separately from trigger-derived AequilibraE `distance`.
- [x] 4.4 Decide whether source-ID preservation uses importer-added core-table columns or a separate source-mapping table.
- [x] 4.5 Decide where supported link-count associations are stored durably, if any, and keep unsupported count-location fields diagnostic-only.
- [x] 4.6 Add SQL schema and migration artifacts for link-count or source-mapping storage if required.
- [x] 4.7 Add tests for any new table list, trigger list, migration listing, metadata documentation, and project upgrade behavior.
- [x] 4.8 Complete AequilibraE integration reviewer checkpoint B before coding schema-backed storage.

## 5. Fixtures And Test-First Coverage

- [x] 5.1 Add compact VISUM-like GeoJSON fixtures covering nodes, links, zone centroids, optional zone polygons, connectors, count locations, asymmetric directions, unmapped values, invalid units, missing CRS, and deferred public-transport layers.
- [x] 5.2 Add tests for field inventory and layer discovery before implementing importer behavior.
- [x] 5.3 Add tests for default and user-overridden mode mappings, including `CAR` and `HGV` behavior.
- [x] 5.4 Add tests for default and user-overridden link-type mappings and fallback behavior.
- [x] 5.5 Add tests for CRS handling with declared, user-supplied, missing, and explicitly accepted default CRS.
- [x] 5.6 Add tests for unit parsing for length, speed, time, and capacity fields, including unparsable values.
- [x] 5.7 Add tests for topology validation failures: missing node references, endpoint mismatch, missing zone centroid references, and connector mismatch.
- [x] 5.8 Add tests for assignment-readiness failures: missing, zero, negative, NaN, or unparsable time and capacity values.
- [x] 5.9 Add tests that supported link-count associations are preserved even when not usable for assignment validation.
- [x] 5.10 Add tests that count locations do not modify OD matrices, trigger demand adjustment, or perform calibration.
- [x] 5.11 Add tests that deferred public-transport layers are reported and not imported into private-traffic network tables.

## 6. Reader And Validation Implementation

- [x] 6.1 Implement layer discovery for conventional folder file names and explicit file-path handling for VISUM GeoJSON inputs.
- [x] 6.2 Implement GeoPandas-based layer reading with CRS validation and optional reprojection.
- [x] 6.3 Implement field inventory generation for diagnostics and reviewer workflows.
- [x] 6.4 Implement unit parsers for VISUM length, speed, time, and capacity-like fields.
- [x] 6.5 Implement mapping configuration loading, default mapping application, and user override validation.
- [x] 6.6 Implement node/link source-reference validation and link geometry endpoint compatibility checks.
- [x] 6.7 Implement zone centroid, zone polygon, and connector source-reference validation and connector geometry compatibility checks.
- [x] 6.8 Implement count-location source-reference validation for link-count associations and diagnostics for deferred count-location fields.
- [x] 6.9 Implement recognized-deferred-layer reporting for stops, stop points, line routes, public-transport-only layers, and OD matrix files.
- [x] 6.10 Complete reviewer checkpoint C for diagnostics, validation severity, and public API behavior.

## 7. Network Import Implementation

- [x] 7.1 Implement the public network API entry point and keep behavior consistent with existing `Network.create_from_osm(...)` and `Network.create_from_gmns(...)` patterns.
- [x] 7.2 Implement mode and link-type creation or validation before any link insert, including default creation/validation of `h` when `HGV` maps to a separate heavy-vehicle mode.
- [x] 7.3 Import VISUM nodes while preserving source node identifiers and geometry.
- [x] 7.4 Import VISUM zones and centroids while preserving source zone identifiers and geometry.
- [x] 7.5 Import VISUM links with AequilibraE direction, modes, link types, geometry, assignment fields, and source metadata.
- [x] 7.6 Derive assignment-ready numeric fields for distance, speed, capacity, and free-flow time where possible.
- [x] 7.7 Preserve source VISUM length separately when configured and when it differs from geometry-derived distance.
- [x] 7.8 Import VISUM connectors as centroid connectors with configured modes, link type, geometry, and source metadata.
- [x] 7.9 Import or report supported link-count associations without performing demand adjustment.
- [x] 7.10 Ensure failed imports are transactional or provide explicit partial-import diagnostics and cleanup guidance.

## 8. Graph, Assignment, And Database Verification

- [x] 8.1 Test trigger behavior after import: `a_node`, `b_node`, `distance`, node `modes`, and node `link_types`.
- [x] 8.2 Verify imported centroids with `Network.count_centroids()` and graph centroid arrays.
- [x] 8.3 Verify imported connectors connect centroid nodes to regular network nodes and support configured private modes.
- [x] 8.4 Run `project.network.build_graphs(fields=[...], modes=["c"])` on imported fixtures.
- [x] 8.5 Verify `TrafficAssignment.set_time_field(...)` and `set_capacity_field(...)` pass for configured assignment-ready imported fields.
- [x] 8.6 Verify supported link-count associations can be queried with network associations intact.
- [x] 8.7 Complete reviewer checkpoint D before documenting the workflow as assignment-ready.

## 9. UI, Documentation, And Examples

- [x] 9.1 Add or update UI wiring so users can launch the VISUM GeoJSON import workflow from the existing frontend when that frontend lives in this repository.
- [x] 9.2 Add API docstrings for the VISUM GeoJSON import entry point, mapping configuration, and diagnostics/report object.
- [x] 9.3 Add user documentation for required layers, optional zone polygons, CRS handling, default mappings, assignment readiness, UI access, link-count scope, and deferred workflows.
- [x] 9.4 Add a runnable documentation example using compact local fixtures and no external downloads.
- [x] 9.5 Document count locations as future inputs for validation, calibration, and ODME workflows, not as v1 demand-estimation logic.
- [x] 9.6 Complete reviewer checkpoint E for deferred public transport, OD matrix import, and count-based OD adjustment.

## 10. Final Verification

- [x] 10.1 Run focused project/network importer tests.
- [x] 10.2 Run focused graph and traffic-assignment tests against imported VISUM-like fixtures.
- [x] 10.3 Run migration/schema tests if count-location or source-mapping storage changes the project database.
- [x] 10.4 Run relevant documentation doctests or example checks.
- [x] 10.5 Run `ruff check aequilibrae/` on touched package code.
- [x] 10.6 Run `openspec.cmd status --change add-visum-geojson-import-pipeline` and resolve any incomplete artifacts.

## 11. Post-Review Real VISUM Connector Compatibility

- [x] 11.1 Document that connector `NO` is optional and deterministic connector keys are generated from zone, node, and usable direction.
- [x] 11.2 Implement connector import for layers without a numeric `NO`, preserving optional `visum_connector_no` and durable `visum_connector_key`.
- [x] 11.3 Add regression tests for connector layers without `NO` and for duplicate deterministic connector keys.
- [x] 11.4 Run focused VISUM importer tests and linting for touched files.

## 12. Explicit Transport-System Mapping Decisions

- [x] 12.1 Document that transport systems outside the default mapping require explicit mapping or explicit ignore decisions.
- [x] 12.2 Add an `ignored_transport_systems` API option and report unmapped/ignored transport systems with structured diagnostics.
- [x] 12.3 Skip records whose transport systems are all explicitly ignored and keep imported counts consistent.
- [x] 12.4 Add regression tests for extra transport systems, explicit ignores, and user mappings such as `BUS -> t`.
- [x] 12.5 Run focused VISUM importer tests and linting for touched files.

## 13. Numeric Link-Type Naming Compatibility

- [x] 13.1 Document that numeric VISUM `TYPENO` fallback values produce distinct AequilibraE link-type names.
- [x] 13.2 Preserve digits in generated link-type names while keeping valid names for values that begin with digits.
- [x] 13.3 Add regression tests for numeric `TYPENO` values such as `2` and `92`.
- [x] 13.4 Run focused VISUM importer tests and linting for touched files.

## 14. Coincident VISUM Node Compatibility

- [x] 14.1 Document that coincident VISUM nodes can encode separate modal/topological layers and must not be merged by default.
- [x] 14.2 Add a `duplicate_node_policy` API option with deterministic offset behavior and strict error behavior.
- [x] 14.3 Preserve original VISUM node coordinates and offset diagnostics when disambiguating coincident nodes.
- [x] 14.4 Update link and connector geometries to use adjusted endpoint coordinates for offset nodes.
- [x] 14.5 Add regression tests for duplicate-node offset import and strict duplicate-node rejection.

## 15. Source Node And Zone ID Collision Compatibility

- [x] 15.1 Document that VISUM regular node numbers may collide with VISUM zone numbers.
- [x] 15.2 Preserve zone IDs as centroid node IDs and remap only conflicting regular network nodes to free AequilibraE node IDs.
- [x] 15.3 Preserve `visum_node_no` and source-reference mappings for remapped regular nodes.
- [x] 15.4 Update link and connector endpoint imports to use the source-to-AequilibraE node ID mapping.
- [x] 15.5 Add regression tests for a VISUM regular node whose `NO` collides with a zone centroid `NO`.

## 16. Coincident Zone Centroid Compatibility

- [x] 16.1 Document that VISUM zone centroids may share coordinates with regular network nodes.
- [x] 16.2 Offset coincident zone centroid node geometries while preserving original VISUM centroid coordinates.
- [x] 16.3 Update connector geometries to start from adjusted centroid coordinates.
- [x] 16.4 Add regression tests for coincident zone centroid offset behavior.
- [x] 16.5 Normalize adjusted VISUM link and connector endpoint geometries to AequilibraE XY line geometry.

## 17. Compact Imported Link IDs

- [x] 17.1 Document that imported AequilibraE link IDs are compact internal IDs while VISUM source IDs are preserved separately.
- [x] 17.2 Update VISUM link and connector import to allocate compact `links.link_id` values.
- [x] 17.3 Update source-reference and count-location mappings to point from VISUM source IDs to compact imported link IDs.
- [x] 17.4 Add regression tests for high sparse VISUM link numbers and compact imported link IDs.
- [x] 17.5 Regenerate the Karlsruhe example project from Downloads GeoJSON files and import `Visum_3_modes.omx`.
- [x] 17.6 Run focused VISUM importer, matrix import, and assignment smoke checks.

## 18. Connector Assignment Defaults

- [x] 18.1 Document fallback connector travel-time and capacity behavior when VISUM GeoJSON lacks connector assignment fields.
- [x] 18.2 Implement deterministic connector fallback speed and high capacity defaults in the VISUM importer.
- [x] 18.3 Add regression tests for connector fallback assignment values and diagnostics.
- [x] 18.4 Regenerate the Karlsruhe example project and rerun assignment-readiness checks.
