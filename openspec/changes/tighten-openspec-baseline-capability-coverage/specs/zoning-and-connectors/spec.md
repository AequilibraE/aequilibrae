## ADDED Requirements

### Requirement: Project zoning is available

The system SHALL expose project zoning data through the loaded project.

#### Scenario: Accessing zoning data

- **WHEN** a project is opened
- **THEN** the system SHALL provide access to zone records stored in the project database
- **AND** preserve zone identifiers and geometries as project data

### Requirement: Zones and centroids are linked concepts

The system SHALL support zone and centroid workflows that connect traffic analysis areas to network nodes.

#### Scenario: Using centroid nodes

- **WHEN** zones are used for graph or connector workflows
- **THEN** the system SHALL treat centroid nodes as the network representation of traffic analysis zones where applicable

### Requirement: Closest-zone lookup is supported

The system SHALL support lookup of the closest zone for geospatial inputs.

#### Scenario: Finding a zone for a point

- **WHEN** closest-zone lookup is requested for a point geometry
- **THEN** the system SHALL identify the nearest applicable project zone according to the zoning implementation

### Requirement: Bulk centroid connector creation is supported

The system SHALL create centroid connector links between centroid nodes and eligible network nodes.

#### Scenario: Creating connectors

- **WHEN** bulk connector creation is requested with modes and connector limits
- **THEN** the system SHALL create connector links with centroid endpoints, supported modes, valid geometries, and centroid connector link type

### Requirement: Connector creation avoids duplicate work

The system SHALL avoid creating duplicate connectors when existing connectors already satisfy the requested work.

#### Scenario: Re-running connector creation

- **WHEN** connector creation is run again with the same effective inputs
- **THEN** the system SHALL preserve existing suitable connectors rather than adding duplicates

