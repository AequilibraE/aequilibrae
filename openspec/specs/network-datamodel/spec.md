# Network Data Model Specification

## Purpose

This specification captures the current behavioral contract for AequilibraE network data stored in SQLite/SpatiaLite project databases.

## Requirements

### Requirement: Network tables are initialized

The system SHALL initialize the network database using the ordered SQL table list and trigger list shipped with the package.

#### Scenario: Building network schema

- **WHEN** a new project or scenario network database is initialized
- **THEN** the system SHALL create all tables listed in the network `table_list.txt`
- **AND** create all triggers listed in the network `triggers_list.txt`
- **AND** mark existing schema migrations as seen for a newly created database

### Requirement: Links represent network edges

The system SHALL store network links with stable identifiers, node endpoints, direction, modes, link type, directional attributes, and geometry.

#### Scenario: Storing a link

- **WHEN** a link is persisted in the network database
- **THEN** it SHALL have a unique positive `link_id`
- **AND** it SHALL reference `a_node` and `b_node`
- **AND** it SHALL have a direction limited to `-1`, `0`, or `1`
- **AND** it SHALL have at least one allowed mode
- **AND** it SHALL store its geometry as a SpatiaLite `LINESTRING`

### Requirement: Nodes represent network vertices

The system SHALL store network nodes with stable identifiers, centroid flags, derived connectivity metadata, and geometry.

#### Scenario: Storing a node

- **WHEN** a node is persisted in the network database
- **THEN** it SHALL have a unique `node_id`
- **AND** it SHALL indicate whether it is a centroid
- **AND** it SHALL store its geometry as a SpatiaLite `POINT`

### Requirement: Zones represent traffic analysis areas

The system SHALL store zones as geospatial traffic analysis areas.

#### Scenario: Storing a zone

- **WHEN** a zone is persisted in the network database
- **THEN** it SHALL have a unique `zone_id`
- **AND** it SHALL store its geometry as a SpatiaLite `MULTIPOLYGON`
- **AND** the database triggers SHALL maintain the zone area where applicable

### Requirement: Modes are single-letter identifiers

The system SHALL identify available network modes with single-letter mode IDs and associated assignment metadata.

#### Scenario: Creating default modes

- **WHEN** network tables are initialized
- **THEN** the system SHALL create default modes for car, transit, walk, and bicycle

#### Scenario: Validating mode identifiers

- **WHEN** a mode is stored
- **THEN** its `mode_id` SHALL be exactly one character

### Requirement: Triggers maintain network integrity

The system SHALL use database triggers to maintain spatial and relational consistency for network editing.

#### Scenario: Editing through supported paths

- **WHEN** links, nodes, zones, modes, link types, periods, or scenarios are edited through the supported database/API paths
- **THEN** triggers SHALL maintain derived fields and consistency constraints defined by the database specification

#### Scenario: Changing schema behavior

- **WHEN** a change modifies tables, triggers, protected fields, or required fields
- **THEN** the change SHALL update the SQL specification and tests that validate table and trigger registration

### Requirement: Network import and export is supported

The system SHALL support creating and exporting network data through supported external formats and services.

#### Scenario: Importing from OSM

- **WHEN** a new empty network is created from OpenStreetMap
- **THEN** the system SHALL download applicable network data for the requested area or place
- **AND** populate network links and nodes according to configured modes and OSM parameters

#### Scenario: Importing or exporting GMNS

- **WHEN** GMNS import or export is requested
- **THEN** the system SHALL map configured GMNS fields to or from AequilibraE network fields

