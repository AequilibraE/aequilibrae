# Transit GTFS Specification

## Purpose

This specification captures the current behavioral contract for GTFS import, transit databases, transit graph creation, and transit assignment support.

## Requirements

### Requirement: Transit database is available

The system SHALL ensure that every loaded project has a transit database available for public transport workflows.

#### Scenario: Loading transit gateway

- **WHEN** a project loads its transit gateway
- **THEN** the system SHALL create `public_transport.sqlite` if it does not exist
- **AND** initialize transit tables and triggers

### Requirement: GTFS route systems can be built

The system SHALL create GTFS route-system builders configured for the active project and agency.

#### Scenario: Creating a GTFS builder

- **WHEN** a GTFS builder is requested with agency, file path, day, and description
- **THEN** the system SHALL return a builder configured with default transit capacities and passenger-car equivalents
- **AND** connect builder progress signals to the transit gateway

### Requirement: Transit tables store imported service data

The system SHALL store imported transit service data in the transit database using the transit table specification.

#### Scenario: Storing route data

- **WHEN** GTFS route data is imported
- **THEN** the system SHALL store route patterns, agency references, route metadata, capacities, passenger-car equivalents, and route geometry

#### Scenario: Storing trip schedules

- **WHEN** GTFS trip schedules are imported
- **THEN** the system SHALL store trip IDs, sequence numbers, arrivals, and departures with trip relationships

### Requirement: Transit graphs are period-aware

The system SHALL create, save, remove, and load transit graphs by project network period.

#### Scenario: Creating a transit graph

- **WHEN** a transit graph is created without an explicit period
- **THEN** the system SHALL use the project's default period
- **AND** store the graph in the transit graph registry for that period

#### Scenario: Loading saved transit graphs

- **WHEN** saved transit graphs are loaded
- **THEN** the system SHALL load graph configurations for requested periods or all available periods

### Requirement: Transit preload can be computed

The system SHALL compute transit preload vectors over a requested time window.

#### Scenario: Computing preload

- **WHEN** transit preload is requested with start and end times
- **THEN** the system SHALL aggregate transit trips active in the requested window into link-direction preload values

#### Scenario: Selecting inclusion condition

- **WHEN** an inclusion condition is specified
- **THEN** the system SHALL include trips according to start, end, midpoint, or any-stop schedule logic as supported

### Requirement: Transit assignment is supported

The system SHALL support public transport assignment through transit classes and Optimal Strategies workflows.

#### Scenario: Executing transit assignment

- **WHEN** a transit assignment is configured with transit classes and required graph data
- **THEN** the system SHALL execute the selected transit assignment procedure
- **AND** expose assignment results through the assignment result interfaces

