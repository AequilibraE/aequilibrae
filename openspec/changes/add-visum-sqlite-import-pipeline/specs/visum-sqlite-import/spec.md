## ADDED Requirements

### Requirement: VISUM SQLite imports private-traffic network objects
The system SHALL import VISUM SQLite private-traffic network objects into an AequilibraE project using deterministic
source-table mappings.

#### Scenario: Importing required SQLite object tables
- **WHEN** a VISUM SQLite import is requested with `NODE`, `LINK`, `CONNECTOR`, `ZONE`, `TSYS`, and `LINKTYPE` tables
- **THEN** the system SHALL read the source tables from the supplied SQLite database
- **AND** create AequilibraE nodes, links, zones, centroids, connectors, modes, and link types according to configured
  VISUM mappings
- **AND** preserve VISUM source identifiers needed to trace imported records to the source tables

#### Scenario: Rejecting missing required SQLite tables
- **WHEN** a VISUM SQLite import is requested without a required source table
- **THEN** the system SHALL reject the import with diagnostics that identify the missing table

#### Scenario: Reporting deferred VISUM SQLite tables
- **WHEN** VISUM public-transport schedule, stop timing, vehicle journey, route-system, turn, fare, point-of-interest, or
  other unsupported source tables are present
- **THEN** the system SHALL report that those tables are recognized but not imported into the private-traffic network in
  this version

### Requirement: VISUM SQLite topology is preserved from source identifiers
The system SHALL construct network topology from VISUM SQLite source identifiers rather than inferred spatial snapping.

#### Scenario: Collapsing directed link rows
- **WHEN** two VISUM `LINK` rows share a source `NO` and represent opposite `FROMNODENO`/`TONODENO` directions
- **THEN** the system SHALL import one AequilibraE link with directional AB/BA mode and assignment attributes
- **AND** assign a compact internal AequilibraE `link_id`
- **AND** preserve the VISUM source link `NO` separately

#### Scenario: Skipping source links with no usable transport systems
- **WHEN** both VISUM SQLite directions for a source link have no declared transport systems after configured mapping and
  ignore rules
- **THEN** the system SHALL skip that source link
- **AND** report diagnostics that identify the skipped VISUM source link

#### Scenario: Importing connectors from source keys
- **WHEN** a VISUM SQLite connector is identified by `ZONENO`, `NODENO`, and `DIRECTION`
- **THEN** the system SHALL import the connector as a centroid connector associated with the referenced zone and node
- **AND** preserve a deterministic connector source key for traceability

#### Scenario: Preserving coincident node topology
- **WHEN** multiple VISUM SQLite `NODE` records share identical coordinates
- **THEN** the system SHALL NOT merge the source nodes by default
- **AND** SHALL preserve separate source node identifiers with deterministic coordinate offset behavior
- **AND** SHALL report the duplicate coordinate group and applied offset diagnostics

#### Scenario: Handling source node and zone ID collisions
- **WHEN** a VISUM SQLite regular node `NO` collides with a VISUM SQLite zone `NO`
- **THEN** the system SHALL preserve the zone ID as the AequilibraE centroid node ID
- **AND** import the regular source node with a deterministic free AequilibraE node ID
- **AND** import links and connectors using the mapped AequilibraE regular node ID

### Requirement: VISUM SQLite geometry is reconstructed faithfully
The system SHALL reconstruct AequilibraE geometries from VISUM SQLite coordinate and relational geometry tables without
inventing geometry not represented by the source export.

#### Scenario: Reconstructing link geometry
- **WHEN** a VISUM SQLite source link has ordered vertices in `LINKPOLY`
- **THEN** the system SHALL construct link geometry from the source node endpoint, ordered `LINKPOLY` vertices, and target
  node endpoint
- **AND** use the reverse vertex order for the opposite source direction when needed

#### Scenario: Using straight link geometry when no link polyline exists
- **WHEN** a VISUM SQLite source link has no `LINKPOLY` vertices in either direction
- **THEN** the system SHALL construct straight node-to-node geometry from the source node coordinates
- **AND** report this as source-straight geometry rather than a fallback error

#### Scenario: Reconstructing connector geometry
- **WHEN** a VISUM SQLite connector references a zone centroid and network node
- **THEN** the system SHALL construct straight centroid-to-node connector geometry unless a supported source connector
  geometry table is available

#### Scenario: Reconstructing zone polygons
- **WHEN** a VISUM SQLite zone references a reconstructable `SURFACEID`
- **THEN** the system SHALL reconstruct zone geometry from the related surface, face, edge, edge-item, and point tables
- **AND** store supported geometry as an AequilibraE zone multipolygon

### Requirement: VISUM SQLite CRS handling is explicit
The system SHALL transform VISUM SQLite projected coordinates into the project storage CRS using explicit source CRS
metadata.

#### Scenario: Reading source projection definition
- **WHEN** `NETWORK.PROJECTIONDEFINITION` contains a parseable CRS definition
- **THEN** the system SHALL parse it with the project CRS tooling
- **AND** include the parsed source CRS in import diagnostics

#### Scenario: Rejecting missing or unparseable CRS
- **WHEN** VISUM SQLite source coordinates are projected but the source CRS is missing or unparseable
- **THEN** the system SHALL require a caller-supplied source CRS or explicit default acceptance before writing network
  records

### Requirement: VISUM SQLite mappings are deterministic and configurable
The system SHALL map VISUM SQLite transport systems, modes, link types, and fields using deterministic defaults and
caller-provided overrides.

#### Scenario: Mapping transport systems from TSYSSET
- **WHEN** VISUM SQLite `LINK` or `CONNECTOR` rows declare `TSYSSET` values
- **THEN** the system SHALL map only configured transport systems into single-character AequilibraE mode identifiers
- **AND** report unmapped transport systems before import writes to the project database

#### Scenario: Reading VISUM system definitions
- **WHEN** VISUM SQLite `TSYS` or `MODE` definitions are present
- **THEN** the system SHALL use them for diagnostics and validation of configured transport-system mappings
- **AND** SHALL NOT require runtime AI or heuristic mode inference

#### Scenario: Mapping link types
- **WHEN** VISUM SQLite link `TYPENO`, `LC`, or `LINKTYPE` values are mapped to AequilibraE link types
- **THEN** the system SHALL create or validate corresponding AequilibraE link types according to deterministic configured
  rules

### Requirement: VISUM SQLite assignment fields use source values
The system SHALL derive assignment-ready private-traffic fields from VISUM SQLite source assignment values where
available.

#### Scenario: Deriving link assignment fields
- **WHEN** a VISUM SQLite link direction includes positive length, speed, and capacity values
- **THEN** the system SHALL derive numeric AequilibraE distance, speed, capacity, and free-flow travel-time fields for the
  mapped private-traffic modes

#### Scenario: Using connector transport-system travel times
- **WHEN** a VISUM SQLite connector includes `T0_TSYS(<transport system>)` for a mapped private-traffic mode
- **THEN** the system SHALL use that source connector travel time for the imported connector direction
- **AND** SHALL NOT replace it with the GeoJSON missing-field fallback behavior

#### Scenario: Handling explicit zero connector travel times
- **WHEN** a VISUM SQLite connector explicitly stores zero travel time for a mapped private-traffic transport system
- **THEN** the system SHALL import a positive epsilon travel time suitable for AequilibraE graph and assignment algorithms
- **AND** report diagnostics identifying the source connector and zero-time field

#### Scenario: Reporting non-assignment-ready SQLite records
- **WHEN** a mapped private-traffic SQLite link or connector lacks required positive capacity or computable time values
  after source-value and epsilon rules are applied
- **THEN** the system SHALL report the affected source records and fields
- **AND** indicate whether graph building can proceed without assignment readiness

### Requirement: VISUM SQLite graph connectivity can be validated
The system SHALL provide tests or validation routines that compare imported AequilibraE graph connectivity against VISUM
SQLite source connectivity by mapped mode.

#### Scenario: Comparing SQLite import to SQLite source connectivity
- **WHEN** a VISUM SQLite project is imported and graph connectivity validation is requested
- **THEN** the system SHALL compare directed node-to-node and centroid-to-node arc availability for each mapped mode
  between the imported AequilibraE graph inputs and the VISUM SQLite source tables
- **AND** report missing, extra, or mode-mismatched arcs

#### Scenario: Comparing GeoJSON import to SQLite source connectivity
- **WHEN** equivalent VISUM GeoJSON and SQLite exports are available for the same source model
- **THEN** the system SHALL support a validation test that compares the GeoJSON-imported modal connectivity against the
  SQLite source modal connectivity
- **AND** distinguish connectivity differences from expected impedance differences caused by fallback or epsilon values

#### Scenario: Comparing SQLite and GeoJSON imported connectivity
- **WHEN** equivalent VISUM SQLite and GeoJSON imports have been built as AequilibraE projects
- **THEN** the system SHALL support validation that both imports expose the same mapped-mode link and connector
  connectivity, subject to documented deferred features such as turn restrictions

### Requirement: VISUM SQLite count-location associations are preserved
The system SHALL preserve supported VISUM SQLite count-location associations for validation and future calibration
workflows.

#### Scenario: Associating count locations to imported links
- **WHEN** `COUNTLOCATION` rows reference imported source links through `LINKNO`, `FROMNODENO`, and `TONODENO`
- **THEN** the system SHALL preserve the association needed to compare observations to assigned link flows

#### Scenario: Deferring count-based demand adjustment
- **WHEN** VISUM SQLite count-location fields are imported or reported
- **THEN** the system SHALL NOT adjust OD matrices or traffic demand as part of the SQLite network import pipeline
