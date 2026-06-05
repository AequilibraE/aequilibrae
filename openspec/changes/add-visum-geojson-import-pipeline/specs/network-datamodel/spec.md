## MODIFIED Requirements

### Requirement: Network import and export is supported
The system SHALL support creating and exporting network data through supported external formats and services.

#### Scenario: Importing from OSM
- **WHEN** a new empty network is created from OpenStreetMap
- **THEN** the system SHALL download applicable network data for the requested area or place
- **AND** populate network links and nodes according to configured modes and OSM parameters

#### Scenario: Importing or exporting GMNS
- **WHEN** GMNS import or export is requested
- **THEN** the system SHALL map configured GMNS fields to or from AequilibraE network fields

#### Scenario: Importing private traffic network data from VISUM GeoJSON
- **WHEN** VISUM GeoJSON private-traffic import is requested
- **THEN** the system SHALL populate network links, nodes, zones, centroids, and connectors according to configured VISUM mappings
- **AND** preserve source identifiers and imported count locations for traceability and validation
- **AND** preserve separate VISUM nodes that share coordinates unless strict duplicate-node handling is requested
- **AND** preserve original VISUM coordinates when imported node geometries are offset to satisfy AequilibraE node uniqueness rules
- **AND** preserve source-to-imported node ID mappings when VISUM regular node numbers collide with centroid node IDs
- **AND** preserve zone centroids with adjusted coordinates when VISUM centroid coordinates collide with another imported node
