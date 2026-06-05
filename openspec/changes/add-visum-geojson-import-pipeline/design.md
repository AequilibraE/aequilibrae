## Context

AequilibraE currently supports OSM and GMNS network import, plus lower-level APIs that can populate links, nodes, zones, modes, and link types from custom layer-processing scripts. VISUM can export rich GeoJSON layer sets with explicit node IDs, link endpoint references, zone centroids, zone polygons, connectors, count locations, and simulator-specific fields. The sample VISUM GeoJSON files inspected during discovery contain exact link-to-node and connector-to-centroid topology, which means the importer can preserve simulator IDs instead of inferring topology spatially.

The first version targets private-traffic model construction and assignment readiness from VISUM GeoJSON layers. OD matrix import is the next stage after the network importer is stable. Public transport layers, count-data calibration, and count-based OD adjustment are important but separate workflows.

## Goals / Non-Goals

**Goals:**

- Provide a first-class VISUM GeoJSON import entry point under the project network API.
- Read VISUM GeoJSON layers through GeoPandas/Shapely so geometry, CRS, tabular fields, and future GIS formats can be handled consistently.
- Import private-traffic nodes, links, zone centroids, connectors, optional zone polygons, and VISUM link-count associations that can be connected to imported links.
- Preserve VISUM source identifiers and selected metadata for traceability.
- Use deterministic default mappings with user-overridable configuration for transport systems, link types, and assignment fields.
- Produce validation diagnostics before and during import so users can resolve missing layers, unmapped values, invalid geometry, or non-assignment-ready attributes.
- Keep imported private-traffic networks suitable for `Network.build_graphs()` and `TrafficAssignment` when required fields are present or derivable.

**Non-Goals:**

- Public transport import from VISUM routes, stops, stop points, or line routes.
- GTFS conversion or integration.
- OD matrix import in this change; it is the next planned stage after network import.
- Count-based OD matrix estimation or correction.
- Runtime GenAI-based field mapping in the importer.
- Fuzzy topology construction by snapping nearby link endpoints.

## Decisions

### Use GeoPandas for ingestion, not a hand-rolled GeoJSON parser

The importer will treat GeoJSON as the initial supported exchange format but should read layers through GeoPandas. GeoPandas is already a dependency and gives consistent access to Shapely geometry, CRS metadata, tabular fields, and reprojection. This also leaves room for future simulator exports in GeoPackage or similar GIS formats without redesigning the importer core.

Alternative considered: parse GeoJSON directly with `json`. This is simpler for one file but would duplicate geometry, CRS, and data-frame logic already provided by the project stack.

### Keep mapping deterministic and configurable

The importer will ship deterministic VISUM defaults and accept a user-provided mapping configuration. Unknown transport systems, link classes, or required fields will produce validation errors or warnings according to the configured strictness. GenAI can be useful outside the core library to suggest a mapping file, but the importer itself must be reproducible, testable, offline, and auditable.

Alternative considered: consult a GenAI service at import time. This was rejected for core behavior because it introduces network dependence, privacy concerns for proprietary models, and non-reproducible mappings.

### Preserve source topology instead of snapping

VISUM link layers include `FROMNODENO` and `TONODENO`, and connectors include `ZONENO` and `NODENO`. The importer will validate that referenced nodes and centroids exist and that geometries match their referenced endpoints where required. It will not create topology by snapping nearby endpoints in the first version.

Alternative considered: spatially match endpoints to nearby nodes. This can hide model errors and would require tolerance decisions that vary by CRS and simulator export settings.

### Preserve coincident VISUM nodes by offsetting, not merging

Real VISUM models can contain multiple node records at the same physical coordinates when the records represent distinct
modal or topological layers. For example, one coincident node can serve bus/train/tram links while another serves only
rail-like links. AequilibraE's network triggers reject inserting two nodes at the same coordinates and AequilibraE does
not currently model turn restrictions that would make an automatic merge safe in all cases.

The importer will therefore preserve separate VISUM node IDs by default and apply a tiny deterministic coordinate offset
to all but one node in each duplicate coordinate group. The original VISUM longitude/latitude and optional projected
`XCOORD`/`YCOORD` values are preserved on the imported node, and diagnostics identify the duplicate group and offset.
Imported link and connector geometries are rewritten at their endpoints to match the adjusted node geometry so the
AequilibraE spatial triggers keep the intended source topology. Strict callers can request an error policy instead.

Alternative considered: automatically merge coincident nodes. This was rejected as the default because it can create
artificial connectivity across modal layers, especially while turn restrictions are outside the importer scope.

### Preserve centroid node IDs when source node and zone IDs collide

AequilibraE zoning and connector helpers commonly assume a zone centroid's `node_id` matches the corresponding
`zone_id`. Real VISUM exports may still contain a regular node whose `NO` equals a zone centroid `NO`. In that case, the
importer will keep the zone/centroid ID unchanged and remap the regular source node to a deterministic free AequilibraE
node ID. Links and connectors are inserted through the source-to-imported node mapping, and `nodes.visum_node_no`
preserves the original VISUM number.

Alternative considered: remap centroid node IDs. This was rejected because it would break more existing AequilibraE
assumptions around zone-centroid identity than remapping regular imported nodes.

### Offset zone centroids when their coordinates collide

VISUM zone centroids can share coordinates with regular network nodes. Because AequilibraE stores centroids in the same
`nodes` table and rejects duplicate node geometries, the importer will offset the centroid node geometry when needed
while keeping the zone ID as the centroid node ID. Connector geometries are rewritten to start from the adjusted centroid
coordinate. Original VISUM centroid coordinates remain preserved on the centroid node for traceability and future
geolocation/audit uses.

### Private-traffic first

The first version will focus on private-traffic network import and assignment-ready fields. VISUM public transport concepts overlap only partially with AequilibraE's current GTFS/transit import model, so public transport should be studied and specified separately.

Alternative considered: import all VISUM layers in one change. This would mix private assignment, transit data modeling, and demand workflows into one large change with high schema and documentation risk.

### HGV is preserved as a separate private mode by default

VISUM `HGV` represents heavy goods vehicles. The default mapping will preserve it as a separate AequilibraE mode, expected to be `h`, rather than silently merging it into car mode `c`. Users may override the mapping to merge `HGV` into `c` when they want a single private-traffic class.

Because AequilibraE link records can only reference existing single-character records in the `modes` table, the importer must create or validate the target mode before inserting any link that uses it. For the default HGV mapping, this means creating or validating mode `h` through the existing project network modes API, with a clear name such as `hgv` or `heavy_goods_vehicle`, before importing links or connectors containing `h`.

Alternative considered: merge `HGV` into `c` by default. This was rejected because it loses truck-specific restrictions, PCE, costs, and future class behavior.

### Zone centroids are operational; zone polygons are optional context

Zone centroids are required for assignment because they become centroid nodes where demand enters and exits the network. Zone polygons provide spatial context, aggregation geometry, and support for connector generation, but they are not strictly required when centroids and connectors are already supplied. If connectors are missing, polygons become more valuable but still do not replace the need to connect centroids to the network.

Alternative considered: require both zone centroid and zone polygon layers. This was rejected because VISUM exports can provide enough assignment topology through centroids plus connectors.

### Link counts are the first supported traffic-data object

VISUM count locations should not be modeled as source-shaped geometry tables. For the first private-traffic assignment workflow, the supported traffic-data object is a link count associated with an imported link. The sample VISUM `countlocation` layer contains link association fields (`LINKNO`, `FROMNODENO`, `TONODENO`), location fields (`XCOORD`, `YCOORD`), identity/classification fields (`NO`, `CODE`, `NAME`, `TYPE`, `TYPENO`), time-window fields (`DATE`, `FROM`, `TO`, `LENGTH`), and count-value candidates such as `CAR_ORIG`, `HVG_ORIG`, `MOTOR_ORIG`, and `DTVW`.

The importer should treat observed link-count fields as candidates for validation and future calibration workflows, but it must not perform demand calibration or OD adjustment in this change. Turn-count-like fields in the sample (`CARS_LEFT`, `CARS_RIGHT`, `CARS_STRAIGHT`) and projected fields (`CARS_PROJ`, `HVG_PROJ`, `MOTOR_PROJ`) should be inventoried and reported, but durable support for turn counts, lane/detector counts, screenlines, speeds, routes, and travel-time observations is deferred.

Alternative considered: add a generic VISUM count-location table immediately. This was rejected because count data sources vary widely and a source-shaped table could make future traffic-data workflows harder to generalize.

### Support conventional folder input and explicit file mapping

The public API should support folder-based import for conventional VISUM GeoJSON exports and explicit file mapping for custom names or partial exports. Folder mode should discover common names such as `node.geojson`, `link.geojson`, `connector.geojson`, and `zone_centroid.geojson`. Explicit mode should let users match arbitrary file paths to the layer type they provide.

Alternative considered: require a fixed media folder or copy source files into the project by default. This is not needed for v1; diagnostics and source paths are enough unless later reproducibility requirements justify optional source-file archiving.

Source-file paths will be used in diagnostics during import, but they will not be recorded durably or copied into a project media folder in v1.

## Implementation Review Model

Implementation should use focused reviewer roles before coding the importer body and again before marking the change complete. Draft reviewer charters live in `reviewers.md`.

### VISUM semantics review

The VISUM reviewer confirms the source-layer contract: required layers, field meanings, directional `R_*` semantics, transport-system values, link classifications, connector semantics, count-location references, and deferred public-transport layers. This review must produce or approve a field inventory and a mapping table before implementation depends on those assumptions.

### AequilibraE integration review

The AequilibraE reviewer confirms that API shape, schema choices, import ordering, source-ID preservation, triggers, modes, link types, zones, centroids, connectors, graph building, and assignment-ready fields remain compatible with existing project behavior. This review is mandatory before schema/migration work and before the importer writes records into the project database.

### Traffic data and counts review

The traffic-data reviewer confirms private-traffic assumptions, HGV/CAR handling, capacity source, speed and free-flow time derivation, directional missing-value behavior, connector mode policy, traffic-data object scope, count-location associations, and explicit deferral of ODME/count-based demand adjustment.

### Review checkpoints

- Checkpoint A: field inventory, mapping table, and open decisions are complete before importer behavior is implemented.
- Checkpoint B: schema and link-count storage decisions are approved before SQL or migration changes are made.
- Checkpoint C: importer diagnostics and validation severity levels are approved before public API behavior is stabilized.
- Checkpoint D: graph-building and assignment-readiness tests pass before docs present the workflow as assignment-ready.
- Checkpoint E: deferred public transport, OD matrix import, and count-based OD adjustment are documented and tested as out of scope.

## Risks / Trade-offs

- CRS ambiguity in VISUM GeoJSON exports -> require CRS configuration or explicit user acceptance when CRS metadata is absent; validate coordinate ranges where practical.
- VISUM fields encode units in strings such as `0.548km` and `110km/h` -> centralize parsers for supported units and fail clearly on unknown unit formats.
- Transport-system mapping can be agency/model specific -> provide deterministic defaults and require explicit overrides for ambiguous values.
- HGV may be merged with car in some modeling practice -> preserve `HGV` as mode `h` by default, keep the mapping configurable, and document any merge override clearly in diagnostics.
- Count locations may not fit an existing database table cleanly -> support only link-count associations in v1, report/defer other traffic-data objects, and avoid source-shaped VISUM tables.
- Existing network triggers compute distances from geometry -> avoid fighting trigger-derived fields; preserve source lengths in separate fields when source values differ from geodesic length.
- Large VISUM exports can be slow if inserted row-by-row -> prefer batch database operations where they preserve trigger behavior and transaction integrity.

## Migration Plan

No migration is required for existing projects unless implementation introduces new durable link-count or source-mapping tables. If schema is extended, add SQL specification, migration logic, and tests so older projects can be upgraded safely.

The importer should only populate empty or explicitly targeted projects by default. Rollback for a failed import should be transaction-based where feasible; otherwise the importer must document partial-import behavior and provide enough diagnostics for cleanup.

## Deferred Questions

- Which non-required VISUM source fields should be preserved by default versus available only through optional field mapping?
- Which traffic-data objects after link counts should be supported first: turn counts, detector/lane counts, screenlines, speeds, routes, travel times, or another observation type?
- Which count-data fields and formats should be supported when the later count-data import/calibration stage is specified?
