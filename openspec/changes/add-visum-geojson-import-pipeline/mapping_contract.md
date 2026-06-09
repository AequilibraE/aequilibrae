# VISUM GeoJSON Mapping Contract

## Public API

The public network API is `Project.network.create_from_visum_geojson(...)`.

Accepted inputs:

- A folder path containing conventional layer names such as `node.geojson`, `link.geojson`, `connector.geojson`,
  `zone_centroid.geojson`, optional `zone_polygon.geojson`, and optional `countlocation.geojson`.
- An explicit mapping of importer layer names to file paths for custom names or partial exports.

The importer requires an empty network by default. Importing into a non-empty project is allowed only when the caller
passes an explicit override; this protects users from accidentally blending two network sources.

Coincident VISUM node coordinates are handled with `duplicate_node_policy`. The default policy, `offset`, preserves
separate source nodes by applying a tiny deterministic coordinate offset and preserving the original VISUM coordinates.
The strict policy, `error`, rejects coincident source nodes before project database writes.

VISUM regular node `NO` values are used as AequilibraE node IDs when possible. If a regular node `NO` collides with a
zone centroid `NO`, the importer keeps the zone ID as the centroid node ID and remaps the regular node to a deterministic
free AequilibraE node ID. The returned `source_references["nodes"]` mapping records the source-to-imported ID relation.

VISUM zone centroid coordinates are also disambiguated when they collide with already imported node coordinates. The
zone ID remains the centroid node ID, connector geometries are adjusted to start at the imported centroid coordinate, and
the original VISUM centroid coordinates are preserved on the centroid node.

## Diagnostics

The importer returns a report object with:

- diagnostics containing severity, code, message, layer, field, and optional source record ID;
- discovered layer paths;
- deferred layer names;
- field inventory;
- mapping choices for modes and link types;
- imported row counts;
- source-to-imported-ID references for nodes, links, zones, connectors, and count associations;
- CRS assumptions and reprojection choices.

Severity levels:

- `error`: missing required layers, unresolved topology references, unmapped required modes/link types, invalid CRS
  handling, invalid geometry, or assignment-required fields that cannot be parsed in strict mode.
- `warning`: missing optional layers, CRS default accepted by the user, deferred layers, deferred count fields, or
  records that can build graphs but are not assignment-ready.
- `info`: discovered layers, mapping choices, imported row counts, and explicit out-of-scope behavior.

## Default Mappings

Transport systems:

- `CAR -> c`
- `HGV -> h`

Users may override `HGV -> c` to merge heavy goods vehicles into car mode. Every mapped mode ID must be a single
character and must be created or already valid before links or connectors are inserted.

Any VISUM transport system outside the configured mapping must be explicitly mapped or explicitly listed in
`ignored_transport_systems`. The importer SHALL fail before database writes when extra transport systems are neither
mapped nor ignored. Records whose available transport systems are all explicitly ignored SHALL be skipped and reported.
Records with no declared transport systems in either direction SHALL also be skipped and reported because they cannot
produce an AequilibraE mode string. This keeps simple `CAR`/`HGV` networks automatic while forcing richer VISUM exports
to receive a deliberate modeling decision.

Link types:

- default class/type values map deterministically to AequilibraE link-type records;
- `LC` is preferred over `TYPENO` for default classification;
- numeric `TYPENO` fallback values generate distinct valid names such as `visum_two` and `visum_nine_two`;
- values that exceed configured IDs receive deterministic fallback IDs from unused ASCII letters;
- if no unused ID exists, import fails with an error diagnostic.

Assignment fields:

- speed is stored as km/h in `speed_ab` and `speed_ba`;
- capacity is stored as vehicles per hour in `capacity_ab` and `capacity_ba`;
- free-flow time is stored in minutes in `travel_time_ab` and `travel_time_ba`;
- if explicit VISUM time is missing, free-flow time is derived from parsed source length and speed;
- if connector time and speed are missing but connector length is available, connector free-flow time is derived from
  length using a deterministic 30 km/h fallback connector speed;
- if connector length is missing or non-positive, fallback connector time derivation uses geodesic geometry length;
- if connector capacity is missing, connector capacity is set to a deterministic high value of 99,999;
- connector fallback assignment values are reported with diagnostics;
- geometry-derived AequilibraE `distance` remains trigger-owned;
- parsed VISUM source length is preserved in `visum_length_ab` and `visum_length_ba` when present.

## Source Metadata

Source identifiers are preserved in importer-added core-table columns:

- `nodes.visum_node_no`
- `nodes.visum_zone_no`
- `nodes.visum_original_lon`
- `nodes.visum_original_lat`
- `nodes.visum_xcoord`, when the source node layer provides projected `XCOORD`
- `nodes.visum_ycoord`, when the source node layer provides projected `YCOORD`
- `nodes.visum_duplicate_coord_group`, when a source node belongs to a duplicate coordinate group
- `nodes.visum_coord_offset_m`, the approximate offset applied during import
- `links.visum_link_no`
- `links.visum_connector_no`, when the source connector layer provides a numeric `NO`
- `links.visum_connector_key`, generated deterministically from connector zone, node, and usable direction
- `links.visum_length_ab`
- `links.visum_length_ba`
- `zones.visum_zone_no`

The AequilibraE `nodes.node_id` value may differ from `nodes.visum_node_no` when the source node number collides with a
zone centroid node ID.

The AequilibraE `links.link_id` value is a compact internal identifier assigned during import. It SHALL NOT reuse VISUM
link or connector identifiers when those source identifiers are sparse or high-valued. VISUM source link numbers are
preserved in `links.visum_link_no`, connector source values are preserved in `links.visum_connector_no` or
`links.visum_connector_key`, and `report.source_references` maps source identifiers to imported compact link IDs.

Centroid nodes can also use `visum_original_lon`, `visum_original_lat`, `visum_xcoord`, `visum_ycoord`,
`visum_duplicate_coord_group`, and `visum_coord_offset_m` when a VISUM zone centroid coordinate is offset.

Connector keys use `connector:{ZONENO}:{NODENO}:{direction}`, where `direction` is `B`, `O`, or `D` according to mapped
mode availability in the forward and reverse connector fields. If multiple connector records produce the same key, the
importer appends a stable numeric suffix such as `:2`.

Supported count-location associations are reported in diagnostics and source-reference maps in v1. No new durable
count-location or generic source-mapping table is required for this change, so no SQL schema or migration artifact is
needed.

## Count Locations

`countlocation` is the first recognized traffic-data input. Supported link-count associations use `LINKNO` with optional
`FROMNODENO` and `TONODENO` checks. Observed fields such as `CAR_ORIG`, `HVG_ORIG`, `MOTOR_ORIG`, and `DTVW` are
reported as count candidates. Turn counts, projected values, detector/lane counts, screenlines, speeds, routes, and
travel-time observations are deferred.

Count locations SHALL NOT modify OD matrices, perform demand adjustment, trigger ODME, or calibrate assignment demand in
this import pipeline.

## Checkpoints

Checkpoint A is approved for the mapping contract above.
Checkpoint B is approved with the decision to avoid new schema/migration artifacts and to use importer-added core-table
columns plus diagnostics for v1 count associations.
