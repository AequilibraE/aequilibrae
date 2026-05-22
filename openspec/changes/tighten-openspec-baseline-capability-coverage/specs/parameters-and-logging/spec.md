## ADDED Requirements

### Requirement: Parameters load from the active project first

The system SHALL load parameters from the active project's `parameters.yml` when an active project is available.

#### Scenario: Loading project parameters

- **WHEN** parameters are requested while a project is active
- **THEN** the system SHALL read parameter values from that project's `parameters.yml`

### Requirement: Package defaults are available without a project

The system SHALL provide package default parameters when no active project parameter file is available.

#### Scenario: Loading default parameters

- **WHEN** parameters are requested without an active project
- **THEN** the system SHALL return a copy of the package default parameter structure

### Requirement: New projects receive parameter files

The system SHALL create a project-local parameter file when initializing a new project.

#### Scenario: Creating a new project

- **WHEN** a new project is created
- **THEN** the system SHALL write `parameters.yml` into the project folder using the default parameter structure

### Requirement: Project logging is configured on open

The system SHALL configure the AequilibraE logger for an opened project.

#### Scenario: Opening a project

- **WHEN** a project is opened
- **THEN** the system SHALL attach project logging so runtime messages can be written to the project log

### Requirement: Project logging is cleaned on close

The system SHALL clean project logging resources when a project is closed.

#### Scenario: Closing a project

- **WHEN** a project is closed
- **THEN** the system SHALL close project log handlers and deactivate project-specific logging state

