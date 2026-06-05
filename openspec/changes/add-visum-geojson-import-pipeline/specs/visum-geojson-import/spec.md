## ADDED Requirements

### Requirement: VISUM GeoJSON imports private-traffic network layers
The system SHALL import VISUM GeoJSON private-traffic network layers into an AequilibraE project using deterministic layer and field mappings.

#### Scenario: Importing required private network layers
- **WHEN** a VISUM GeoJSON import is requested with node, link, zone centroid, and connector layers
- **THEN** the system SHALL read the layers through geospatial data-frame handling
- **AND** create or update AequilibraE nodes, links, zones, centroids, and centroid connectors according to the configured mapping
- **AND** preserve VISUM source identifiers needed to trace imported records to the source layers

#### Scenario: Importing optional zone polygons
- **WHEN** a VISUM GeoJSON import includes zone polygon layers
- **THEN** the system SHALL import supported zone polygon data as zone geometry
- **AND** SHALL allow import to proceed without zone polygons when zone centroids and connectors provide the required assignment topology

#### Scenario: Rejecting missing required layers
- **WHEN** a VISUM GeoJSON import is requested without a required node or link layer
- **THEN** the system SHALL reject the import with a diagnostic that identifies the missing layer

### Requirement: VISUM topology is validated from source identifiers
The system SHALL validate VISUM topology using explicit source identifiers rather than fuzzy spatial snapping.

#### Scenario: Validating link endpoint references
- **WHEN** a VISUM link references `FROMNODENO` and `TONODENO`
- **THEN** the system SHALL require both referenced nodes to exist in the node layer
- **AND** verify that the link geometry endpoints are compatible with the referenced node geometries according to the configured validation tolerance

#### Scenario: Preserving coincident node topology
- **WHEN** multiple VISUM node records share identical coordinates
- **THEN** the system SHALL NOT merge the source nodes by default
- **AND** SHALL preserve separate source node identifiers so modal or topological separation in VISUM is retained
- **AND** SHALL support a deterministic offset policy that stores adjusted AequilibraE node coordinates while preserving original VISUM coordinates for traceability
- **AND** SHALL adjust imported link and connector endpoint geometries consistently with the adjusted node coordinates
- **AND** SHALL report the duplicate coordinate group and applied offset diagnostics

#### Scenario: Strictly rejecting coincident nodes
- **WHEN** a VISUM GeoJSON import is requested with strict duplicate-node handling
- **THEN** the system SHALL reject coincident VISUM node coordinates before writing network records to the project database

#### Scenario: Handling source node and zone ID collisions
- **WHEN** a VISUM regular node `NO` collides with a VISUM zone centroid `NO`
- **THEN** the system SHALL preserve the zone ID as the AequilibraE centroid node ID
- **AND** SHALL import the regular source node with a deterministic free AequilibraE node ID
- **AND** SHALL preserve the original VISUM node number in source metadata and source-reference mappings
- **AND** SHALL import links and connectors using the mapped AequilibraE regular node ID

#### Scenario: Preserving zone centroids that coincide with network nodes
- **WHEN** a VISUM zone centroid shares coordinates with another imported node
- **THEN** the system SHALL preserve the zone ID as the AequilibraE centroid node ID
- **AND** SHALL apply a deterministic coordinate offset to the centroid node geometry
- **AND** SHALL preserve original VISUM centroid coordinates for traceability
- **AND** SHALL adjust imported connector start geometries consistently with the adjusted centroid coordinate
- **AND** SHALL report the applied centroid offset diagnostic

#### Scenario: Validating connector references
- **WHEN** a VISUM connector references `ZONENO` and `NODENO`
- **THEN** the system SHALL require the referenced zone centroid and network node to exist
- **AND** import the connector as a centroid connector associated with the referenced zone and node
- **AND** SHALL NOT require a numeric connector `NO` field when the connector can be identified from its zone, node, and directional availability
- **AND** SHALL preserve a deterministic connector source key for traceability

#### Scenario: Rejecting inferred snapping
- **WHEN** link or connector references cannot be resolved by source identifiers
- **THEN** the system SHALL report the unresolved references
- **AND** SHALL NOT silently connect records by nearest geometry

### Requirement: VISUM mappings are deterministic and configurable
The system SHALL use deterministic default mappings and allow user-provided overrides for VISUM transport systems, link classes, and fields.

#### Scenario: Mapping transport systems
- **WHEN** VISUM transport-system values are mapped to AequilibraE modes
- **THEN** the system SHALL map only configured transport systems into single-character AequilibraE mode identifiers
- **AND** report any unmapped transport-system values encountered in imported private-traffic layers
- **AND** require any transport system outside the default mapping to be explicitly mapped or explicitly ignored before import writes to the project database
- **AND** skip records whose transport systems are all explicitly ignored, with diagnostics that identify the skipped scope
- **AND** skip records with no declared transport systems in either direction, with diagnostics that identify the skipped scope

#### Scenario: Mapping heavy goods vehicles
- **WHEN** VISUM private-traffic transport systems include `HGV`
- **THEN** the default mapping SHALL preserve `HGV` as a separate AequilibraE mode
- **AND** the system SHALL create or validate the mapped AequilibraE mode before importing links or connectors that reference it
- **AND** allow users to override the mapping to merge `HGV` into car mode when desired

#### Scenario: Mapping link types
- **WHEN** VISUM link-class or type values are mapped to AequilibraE link types
- **THEN** the system SHALL create or validate corresponding AequilibraE link types according to the configured mapping
- **AND** report any unmapped link-type values that prevent import

#### Scenario: Avoiding runtime AI mapping
- **WHEN** the importer maps VISUM fields or values
- **THEN** the system SHALL use configured deterministic rules
- **AND** SHALL NOT require a runtime GenAI service to decide mappings

### Requirement: Assignment-ready fields are derived where possible
The system SHALL derive private-traffic assignment fields from VISUM source values when the configured mapping provides enough information.

#### Scenario: Deriving directional network fields
- **WHEN** VISUM link records include directional length, speed, capacity, and direction availability values
- **THEN** the system SHALL derive numeric AequilibraE fields for private-traffic distance, speed, capacity, and free-flow time
- **AND** represent directionality using AequilibraE direction and directional field conventions

#### Scenario: Reporting non-assignment-ready records
- **WHEN** a private-traffic link lacks required positive numeric time or capacity values for assignment
- **THEN** the system SHALL report the affected source records and fields
- **AND** indicate whether graph building can proceed without traffic assignment readiness

### Requirement: VISUM CRS handling is explicit
The system SHALL handle VISUM GeoJSON coordinate reference systems explicitly.

#### Scenario: Importing layers with CRS metadata
- **WHEN** VISUM GeoJSON layers provide CRS metadata or a CRS is supplied by the user
- **THEN** the system SHALL transform geometries to the project network CRS when required before storing them

#### Scenario: Importing layers without CRS metadata
- **WHEN** VISUM GeoJSON layers do not provide CRS metadata
- **THEN** the system SHALL require a user-supplied CRS or explicit acceptance of the importer default
- **AND** include the CRS assumption in the import diagnostics

### Requirement: VISUM link-count associations are preserved
The system SHALL preserve supported VISUM count-location information as link-count associations for validation and future calibration workflows.

#### Scenario: Identifying link-count support
- **WHEN** a VISUM count-location layer is provided
- **THEN** the system SHALL identify which records can be associated to imported links using fields such as `LINKNO`, `FROMNODENO`, and `TONODENO`
- **AND** identify observed link-count candidates such as `CAR_ORIG`, `HVG_ORIG`, `MOTOR_ORIG`, and `DTVW`
- **AND** report records or fields that are unsupported by the current link-count model

#### Scenario: Associating count locations to imported links
- **WHEN** count-location source references match imported network links
- **THEN** the system SHALL preserve the supported association needed to compare observations to assigned link flows

#### Scenario: Deferring other count-location fields
- **WHEN** count-location fields represent turn counts, projected values, detector/lane counts, screenlines, speeds, routes, travel times, or other unsupported traffic-data objects
- **THEN** the system SHALL report those fields as recognized but deferred

#### Scenario: Deferring count-based demand adjustment
- **WHEN** count locations are imported
- **THEN** the system SHALL NOT adjust OD matrices or traffic demand from count data as part of this import pipeline
- **AND** SHALL leave count-based validation, calibration, and OD adjustment to separate workflows

### Requirement: VISUM inputs can be provided by folder or explicit layer paths
The system SHALL support both conventional folder-based VISUM GeoJSON import and explicit file-to-layer mapping.

#### Scenario: Importing from conventional folder names
- **WHEN** a VISUM GeoJSON import is requested for a folder containing conventional layer file names
- **THEN** the system SHALL discover recognized layer files by their conventional names
- **AND** map them to the corresponding importer layer types

#### Scenario: Importing from explicit layer paths
- **WHEN** a VISUM GeoJSON import is requested with explicit file paths for layer types
- **THEN** the system SHALL use those paths regardless of file name
- **AND** validate that each supplied file provides the expected layer type

### Requirement: Deferred VISUM layers are reported
The system SHALL identify VISUM layers that are recognized but outside the private-traffic import scope.

#### Scenario: Encountering public transport layers
- **WHEN** VISUM stop, stop-point, line-route, or public-transport-only layers are present
- **THEN** the system SHALL report that those layers are recognized but not imported into the private-traffic network in this version

#### Scenario: Encountering OD matrix files
- **WHEN** demand matrix files are provided or discovered with the VISUM GeoJSON layers
- **THEN** the system SHALL report that OD matrix import is outside this pipeline unless a separate supported demand import workflow is invoked
