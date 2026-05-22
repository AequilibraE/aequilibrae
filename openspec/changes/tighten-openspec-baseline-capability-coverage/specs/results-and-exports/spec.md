## ADDED Requirements

### Requirement: Project results are registered

The system SHALL maintain a project result registry in the results database.

#### Scenario: Listing project results

- **WHEN** project results are listed
- **THEN** the system SHALL return registered result records available to the project

### Requirement: Assignment results can be persisted

The system SHALL save supported assignment result tables and associated metadata to project result storage.

#### Scenario: Saving assignment results

- **WHEN** supported assignment results are saved for an open project
- **THEN** the system SHALL persist result data and register it for later discovery

### Requirement: Matrix outputs can be registered

The system SHALL register supported matrix output files in the project matrix registry.

#### Scenario: Saving matrix output

- **WHEN** an output matrix is saved through a project-aware workflow
- **THEN** the system SHALL create or update the corresponding matrix registry entry

### Requirement: SimWrapper export generates dashboard files

The system SHALL generate SimWrapper dashboard configuration and supporting output files for a valid AequilibraE project.

#### Scenario: Generating SimWrapper output

- **WHEN** SimWrapper export is requested for a project
- **THEN** the system SHALL write dashboard configuration and supported geospatial/result outputs under the requested output directory

### Requirement: SimWrapper export constrains output location

The system SHALL prevent SimWrapper export from writing files outside the accepted output scope.

#### Scenario: Rejecting unsafe output path

- **WHEN** SimWrapper export is configured with an output path outside the allowed project/output location
- **THEN** the system SHALL reject the path rather than writing files there

### Requirement: aeq-sim exposes SimWrapper generation

The system SHALL provide the `aeq-sim` command entry point for SimWrapper configuration generation.

#### Scenario: Running aeq-sim

- **WHEN** the `aeq-sim` command is invoked with supported arguments
- **THEN** the system SHALL execute the SimWrapper generation workflow for the requested project

