# VISUM GeoJSON Mapping Contract

## Public API

The public network API is `Project.network.create_from_visum_geojson(...)`.

Accepted inputs:

- A folder path containing conventional layer names such as `node.geojson`, `link.geojson`, `connector.geojson`,
  `zone_centroid.geojson`, optional `zone_polygon.geojson`, and optional `countlocation.geojson`.
- An explicit mapping of importer layer names to file paths for custom names or partial exports.

The importer requires an empty network by default. Importing into a non-empty project is allowed only when the caller
passes an explicit override; this protects users from accidentally blending two network sources.

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

Link types:

- default class/type values map deterministically to AequilibraE link-type records;
- `LC` is preferred over `TYPENO` for default classification;
- values that exceed configured IDs receive deterministic fallback IDs from unused ASCII letters;
- if no unused ID exists, import fails with an error diagnostic.

Assignment fields:

- speed is stored as km/h in `speed_ab` and `speed_ba`;
- capacity is stored as vehicles per hour in `capacity_ab` and `capacity_ba`;
- free-flow time is stored in minutes in `travel_time_ab` and `travel_time_ba`;
- if explicit VISUM time is missing, free-flow time is derived from parsed source length and speed;
- geometry-derived AequilibraE `distance` remains trigger-owned;
- parsed VISUM source length is preserved in `visum_length_ab` and `visum_length_ba` when present.

## Source Metadata

Source identifiers are preserved in importer-added core-table columns:

- `nodes.visum_node_no`
- `nodes.visum_zone_no`
- `links.visum_link_no`
- `links.visum_connector_no`
- `links.visum_length_ab`
- `links.visum_length_ba`
- `zones.visum_zone_no`

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
