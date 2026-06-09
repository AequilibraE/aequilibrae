## Why

AequilibraE can now import VISUM GeoJSON exports, but VISUM SQLite exports preserve richer object data such as
connector travel times, transport-system definitions, link-type defaults, and relational geometry. Supporting SQLite
import gives users a more faithful path from VISUM to AequilibraE and provides a durable way to validate whether
GeoJSON imports preserve modal connectivity.

## What Changes

- Add a VISUM SQLite import pipeline for private-traffic network construction.
- Import VISUM `NODE`, `LINK`, `CONNECTOR`, `ZONE`, `TSYS`, `MODE`, `LINKTYPE`, and supported `COUNTLOCATION` data from a
  SQLite export.
- Reconstruct source-faithful link and zone geometries from VISUM relational geometry tables where available.
- Preserve source VISUM identifiers while assigning compact AequilibraE internal link IDs.
- Reuse the VISUM mapping and topology safeguards established for GeoJSON import, including duplicate-node handling,
  zone/node ID collision handling, connector source keys, explicit transport-system mapping, and diagnostics.
- Use SQLite-native assignment values, including connector `T0_TSYS(...)` values, and treat explicit zero connector
  private-traffic times as positive epsilon costs rather than missing exported data.
- Add validation utilities or tests that compare imported AequilibraE graph connectivity by mode against the VISUM
  SQLite source connectivity and against the existing GeoJSON-derived Karlsruhe project.
- Keep public-transport schedule import, turn restrictions, OD matrix import, and count-based demand adjustment outside
  this change.

## Capabilities

### New Capabilities

- `visum-sqlite-import`: VISUM SQLite source discovery, relational object parsing, geometry reconstruction, import
  behavior, source metadata preservation, assignment-field derivation, diagnostics, and connectivity validation.

### Modified Capabilities

- `network-datamodel`: Network import boundaries expand to include VISUM SQLite private-traffic network import.
- `documentation-and-examples`: User-facing documentation and examples describe the VISUM SQLite import workflow and
  validation relationship to VISUM GeoJSON imports.

## Impact

- Adds a public import entry point under the project network API, likely `Project.network.create_from_visum_sqlite(...)`
  or an equivalent method consistent with `create_from_visum_geojson(...)`.
- Touches VISUM import shared mapping/diagnostic code, network table population, mode and link-type creation, zone and
  centroid handling, connector import, graph-building validation tests, and documentation.
- Uses existing SQLite, Shapely, and pyproj dependencies; no new external runtime dependency is expected.
- Adds focused fixtures or local-data tests for VISUM SQLite object tables, relational geometry reconstruction, compact
  IDs, duplicate coordinates, zero connector times, and graph connectivity equivalence.
