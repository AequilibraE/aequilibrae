## Why

AequilibraE can import networks from OSM, GMNS, and manual link-layer workflows, but it does not provide a first-class path for importing rich simulator exports such as PTV VISUM GeoJSON layers. Supporting this workflow would make AequilibraE easier to use with established planning models while preserving enough source metadata to validate, assign, and later calibrate imported models.

## What Changes

- Add a VISUM GeoJSON import pipeline for private-traffic network construction.
- Expose the importer through the existing user interface where network imports are initiated, such as a File menu or equivalent import action.
- Import and validate VISUM node, link, zone centroid, connector, optional zone polygon, and supported count-location information.
- Map VISUM transport systems, link classes, directions, capacities, speeds, and lengths into AequilibraE network fields using deterministic defaults and user-overridable mapping configuration.
- Preserve relevant VISUM source identifiers and metadata so imported AequilibraE records remain traceable to source layers.
- Create assignment-ready graph fields for private traffic when required source values are available.
- Analyze count-location information against AequilibraE traffic-data needs and preserve supported network associations where possible.
- Defer public transport import, OD matrix import, count-data calibration, and count-based OD adjustment to later changes.

## Capabilities

### New Capabilities

- `visum-geojson-import`: VISUM GeoJSON layer discovery, validation, mapping, import behavior, source metadata preservation, and count-location import.

### Modified Capabilities

- `network-datamodel`: Network import/export boundaries expand to include VISUM GeoJSON private-traffic network import.
- `documentation-and-examples`: User-facing documentation and examples must describe the VISUM GeoJSON import workflow.

## Impact

- Adds a public import entry point under the project network API, likely `Project.network.create_from_visum_geojson(...)` or an equivalent named method.
- Adds or updates UI wiring so users can launch the VISUM GeoJSON import workflow from the existing frontend.
- Uses existing GeoPandas/Shapely/PyProj dependencies for reading, validating, and transforming geospatial layers.
- Touches network table population, mode/link-type creation or validation, zone/centroid handling, connector import, and possibly database schema if supported traffic-data/count-location associations need durable storage.
- Adds tests using compact VISUM-like GeoJSON fixtures covering topology, mappings, assignment fields, connectors, zones, and count locations.
- Adds Sphinx documentation and examples for private-traffic import and known deferred VISUM features.
