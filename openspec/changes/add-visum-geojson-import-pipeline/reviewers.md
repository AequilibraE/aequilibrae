# Reviewer Subagent Drafts

These reviewer roles are intended for implementation planning and review. They do not replace the main implementer; they provide focused domain checks before and during coding.

## VISUM Semantics Reviewer

### Responsibilities

- Interpret PTV VISUM GeoJSON layer semantics before importer implementation begins.
- Confirm which VISUM fields are required, optional, directional, derived, preserved-only, or deferred for private-traffic import.
- Review default mappings from VISUM concepts to AequilibraE concepts, especially transport systems, link types, directions, capacities, speeds, lengths, zones, connectors, and supported count-location associations.
- Identify VISUM public-transport fields and layers that must be explicitly deferred rather than partially or incorrectly imported.
- Ensure importer diagnostics describe VISUM-specific problems in terms a transport modeler can understand.

### Required Inputs

- Sample VISUM GeoJSON files: `node.geojson`, `link.geojson`, `connector.geojson`, `zone_centroid.geojson`, `zone_polygon.geojson`, and `countlocation.geojson`.
- Deferred context layers when available: `stop.geojson`, `stoppoint.geojson`, and `lineroute.geojson`.
- Field inventory for each layer, including data types, null counts, unique categorical values, and example values.
- Known VISUM export assumptions: CRS, units, directional-field meaning, and intended private-traffic transport systems.
- Draft AequilibraE mapping configuration.

### Questions Before Coding

- Which VISUM layers are mandatory for v1 private-traffic import?
- Is `node.geojson` authoritative for topology, or should link endpoint geometries ever create nodes?
- Are `FROMNODENO` and `TONODENO` always the authoritative link endpoints?
- How should `TSYSSET` and `R_TSYSSET` be interpreted when they differ by direction?
- Which VISUM transport systems are private-traffic relevant for v1?
- Should `HGV` be mapped to car mode `c`, a separate heavy-vehicle mode such as `h`, or require explicit user configuration?
- How should empty or missing `TSYSSET` and `R_TSYSSET` values affect import direction and mode availability?
- How should `TYPENO`, `R_TYPENO`, `LC`, and `R_LC` map to or be preserved in AequilibraE?
- Which `R_*` fields are true reverse-direction values, and which are copied metadata?
- Which source fields should be used for capacity, speed, distance, free-flow time, allowed modes, and link type?
- Should source VISUM length be preserved separately from AequilibraE trigger-computed geometry distance?
- Are connectors assignment links, centroid connectors, or both?
- Which connector fields determine allowed private modes?
- Must connector geometry connect exactly from zone centroid to network node?
- Should `zone_polygon.geojson` be required, optional, or preserved when present?
- What does `countlocation.geojson` reference: link, node, detector object, screenline, or a combination, and which references are useful to AequilibraE traffic-data workflows?
- Which count-location fields identify direction, time period, count type, and associated network element?
- Which public-transport layers and fields should be recognized and reported as deferred?

### Validation Gates

- A field dictionary exists for every included VISUM layer before importer coding starts.
- Directional field pairs used by the importer have explicit semantics.
- Private-traffic `TSYSSET` and `R_TSYSSET` values in fixtures are mapped or intentionally rejected.
- `TYPENO`, `R_TYPENO`, `LC`, and `R_LC` values in fixtures are mapped, preserved, or intentionally ignored.
- Link endpoint references are validated through `FROMNODENO` and `TONODENO`, not spatial snapping.
- Connector references are validated through `ZONENO` and `NODENO`, not spatial snapping.
- Supported count-location associations can be imported without implying count-data calibration or OD adjustment.
- Deferred public-transport layers are detected and reported without being imported into private-traffic network tables.

## AequilibraE Integration Reviewer

### Responsibilities

- Review that `Project.network` exposes the importer consistently with `create_from_osm(...)`, `create_from_gmns(...)`, and project lifecycle expectations.
- Protect AequilibraE network contracts: `links`, `nodes`, `zones`, `modes`, `link_types`, centroids, connectors, SpatiaLite geometry columns, and triggers.
- Verify imported networks can build graphs through `Network.build_graphs(...)` and can become `TrafficAssignment`-ready when demand is later supplied.
- Review schema and migration decisions if supported traffic-data, count-location, or imported-layer metadata become durable project data.
- Ensure docs and tests cover public API behavior, assignment readiness, and limitations around public transport, OD, and count-based calibration.

### Required Inputs

- OpenSpec artifacts for `add-visum-geojson-import-pipeline`.
- VISUM sample field inventory for node, link, connector, zone centroid, zone polygon, and count-location layers.
- Proposed VISUM-to-AequilibraE mapping config for modes, link types, direction, speed, capacity, length, and source IDs.
- Decision on supported traffic-data object types and count-location association storage.
- Proposed importer API signature and expected diagnostics object/report.
- Compact test fixtures with topology, centroids, connectors, directional fields, and count locations.

### Questions Before Coding

- Should `create_from_visum_geojson(...)` require an empty network like OSM import, or support importing into explicitly targeted non-empty projects?
- Will VISUM source IDs be stored as new columns on core tables or in a separate source-mapping table?
- If VISUM link classes exceed available single-character `link_type_id` values, what is the deterministic fallback?
- Are VISUM connector records always imported as AequilibraE `centroid_connector` links with `a_node = zone centroid` and `b_node = regular network node`?
- Should source length be preserved separately when it differs from trigger-derived geodesic `distance`?
- Which count-location associations should live durably, and should storage require migrations for existing projects?
- What is the minimum assignment-ready promise: graph builds only, or `TrafficAssignment.set_time_field(...)` and `set_capacity_field(...)` pass for mode `c`?

### Validation Gates

- The importer method is documented, stable, typed, and follows existing `Project.network` patterns.
- Any new durable tables or columns include SQL specs, migrations, metadata documentation, and upgrade tests.
- Stored geometries are compatible with the current SRID 4326 SpatiaLite schema.
- Link and node insertion does not fight triggers that derive `a_node`, `b_node`, node modes, link types, and `distance`.
- Every imported link has non-empty `modes`, and every mode and `link_type` exists before link insertion.
- Imported zone centroids become `nodes.is_centroid = 1`, have positive unique IDs, and align with zone IDs where required by AequilibraE workflows.
- Connectors are valid links, support configured private modes, and connect centroid nodes to regular nodes.
- `project.network.build_graphs(fields=[...], modes=["c"])` succeeds and preserves expected centroids.
- Imported assignment fields contain no NaN, zero, or negative values for selected time and capacity fields.
- Supported count-location associations are imported or reported without triggering OD adjustment or calibration behavior.

## Traffic Data And Counts Reviewer

### Responsibilities

- Review private-traffic mode handling, especially `CAR`, `HGV`, and mixed `TSYSSET` / `R_TSYSSET` values.
- Review assignment-readiness requirements for imported links and connectors.
- Define expectations for deriving capacity, speed, distance, and free-flow travel time from VISUM fields.
- Confirm which VISUM fields are required, optional, preserved, or explicitly deferred.
- Review count-location associations with imported links/nodes and determine which traffic-data object types AequilibraE should support first.
- Ensure supported count-location associations are preserved as validation/calibration assets, not consumed by OD adjustment in v1.
- Identify traffic-modeling assumptions that must be configurable rather than hardcoded.

### Required Inputs

- Sample VISUM GeoJSON layers: `link`, `node`, `connector`, `zone_centroid`, `zone_polygon`, and `countlocation`.
- Field inventory for directional VISUM attributes, especially `TSYSSET`, `R_TSYSSET`, `CAP_*`, `R_CAP_*`, `V0PRT`, `R_V0PRT`, `LENGTH`, `R_LENGTH`, `TYPENO`, `R_TYPENO`, `LC`, and `R_LC`.
- AequilibraE graph and assignment field requirements.
- Proposed default mode and link-type mapping configuration.
- Proposed traffic-data object scope and count-location association model.

### Questions Before Coding

- Should default `HGV` handling merge into car mode `c`, create mode `h`, or require explicit user choice?
- Which capacity field is the default for assignment: hourly capacity, daily capacity, or user-selected?
- Are VISUM `LENGTH` fields authoritative, or should AequilibraE geometry-derived distances be primary?
- How should one-way restrictions and directional missing values be interpreted?
- Should links with non-private transport systems be excluded, imported without assignment fields, or preserved as metadata?
- Should connectors inherit private modes from zones, connected links, or importer configuration?
- What is the minimum count-location association: source VISUM reference only, imported link ID, geometry proximity, or node/link pair?
- Should count locations without resolvable associations be imported with warnings or rejected?

### Validation Gates

- Imported private-traffic links have valid direction, modes, link type, geometry, and source identifiers.
- Assignment-enabled links have strictly positive usable capacity and free-flow time in every permitted direction.
- Speed, capacity, distance, and time parsing fails clearly on unknown formats.
- Directional fields do not silently copy AB values to BA unless explicitly configured.
- HGV/CAR policy is visible in the validation report.
- Supported count-location associations preserve the information needed to compare observations with network flows, speeds, routes, or travel times.
- Count-location handling does not trigger OD estimation, OD correction, or demand calibration in v1.
- Deferred ODME and count-calibration behavior is documented and tested as not implemented.
