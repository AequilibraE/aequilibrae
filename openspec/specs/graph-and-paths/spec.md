# Graph And Paths Specification

## Purpose

This specification captures the current behavioral contract for building graphs from network data and computing paths, skims, and related graph outputs.

## Requirements

### Requirement: Graphs are built per mode

The system SHALL build mode-specific graphs from project network links.

#### Scenario: Building graphs from project links

- **WHEN** `Network.build_graphs()` is called
- **THEN** the system SHALL read link data from the project database
- **AND** create one `Graph` per requested or available mode
- **AND** exclude links that do not support the graph mode

#### Scenario: Filtering by area

- **WHEN** a graph is built with a limiting polygon
- **THEN** the system SHALL use the spatial index to limit candidate links to that area

### Requirement: Graph preparation validates structure

The system SHALL validate required graph fields before preparing a graph for computation.

#### Scenario: Preparing a valid graph

- **WHEN** a graph contains `link_id`, `a_node`, `b_node`, and `direction`
- **THEN** the system SHALL prepare directed graph records and forward-star arrays
- **AND** map original node IDs to dense internal indices

#### Scenario: Rejecting invalid direction values

- **WHEN** a graph contains direction values outside `-1`, `0`, or `1`
- **THEN** the system SHALL reject the graph preparation

### Requirement: Directional links are expanded

The system SHALL convert network link direction into directed graph edges for computation.

#### Scenario: Expanding two-way links

- **WHEN** a link allows travel in both directions
- **THEN** the system SHALL create forward and reverse directed graph records

#### Scenario: Mapping directional attributes

- **WHEN** directional fields exist with `_ab` and `_ba` suffixes
- **THEN** the system SHALL map those fields to the corresponding directed graph records

### Requirement: Centroids are handled explicitly

The system SHALL preserve and prioritize centroid nodes during graph preparation.

#### Scenario: Preparing centroids

- **WHEN** centroids are provided for graph preparation
- **THEN** the system SHALL require unique positive integer centroid IDs
- **AND** place centroids first in the internal node index

#### Scenario: Blocking centroid through-flows

- **WHEN** centroid through-flow blocking is enabled
- **THEN** path and assignment procedures SHALL prevent flows from passing through centroids except as origins or destinations

### Requirement: Graph compression preserves path behavior

The system SHALL support compressed graph representations for efficient skimming and assignment while preserving required centroid behavior.

#### Scenario: Preparing a graph with centroids

- **WHEN** a graph is prepared with centroids
- **THEN** the system SHALL build compressed graph structures for assignment and skimming
- **AND** preserve centroid nodes in the compressed graph

### Requirement: Paths and skims are computed from prepared graphs

The system SHALL compute paths and skims only from graphs prepared for computation.

#### Scenario: Computing a path

- **WHEN** a path is requested between an origin and destination on a prepared graph
- **THEN** the system SHALL return a path result prepared for that graph

#### Scenario: Computing skims

- **WHEN** skimming is requested on a prepared graph with skim fields
- **THEN** the system SHALL compute skim matrices for the configured fields

