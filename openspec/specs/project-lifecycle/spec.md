# Project Lifecycle Specification

## Purpose

This specification captures the current behavioral contract for creating, opening, using, upgrading, and closing AequilibraE projects.

## Requirements

### Requirement: Project creation initializes storage

The system SHALL create a new project only in a non-existing directory and initialize the files needed for normal project operation.

#### Scenario: Creating a new project

- **WHEN** `Project.new(path)` is called with a path that does not already exist
- **THEN** the system SHALL create the project directory
- **AND** create `project_database.sqlite`
- **AND** create project `parameters.yml`
- **AND** create a `run/__init__.py` module
- **AND** initialize the network database tables and triggers

#### Scenario: Rejecting an existing directory

- **WHEN** `Project.new(path)` is called with a path that already exists as a directory
- **THEN** the system SHALL raise an error instead of overwriting that directory

### Requirement: Project opening loads gateways

The system SHALL open an existing project folder by locating its root project database and loading the project gateways used by client code.

#### Scenario: Opening an existing project

- **WHEN** `Project.open(path)` is called for a folder containing `project_database.sqlite`
- **THEN** the system SHALL create the root scenario
- **AND** activate the project as the current project
- **AND** load gateways for network, transit, matrices, results, about metadata, and zoning

#### Scenario: Rejecting a missing project database

- **WHEN** `Project.open(path)` is called for a folder without `project_database.sqlite`
- **THEN** the system SHALL raise an error indicating the model does not exist

### Requirement: Active project is globally available

The system SHALL expose the active project to components that default to project context.

#### Scenario: Activating a project

- **WHEN** a project is opened or created
- **THEN** the system SHALL make it available through the active-project context

#### Scenario: Requesting a missing active project

- **WHEN** code requests the active project and none is active
- **THEN** the system SHALL raise an error unless missing projects are explicitly allowed

### Requirement: Project connections are scoped

The system SHALL expose scoped context managers for the project, transit, and results databases.

#### Scenario: Opening project database connections

- **WHEN** client code enters the project database connection context
- **THEN** the system SHALL provide a SQLite connection to the correct project database
- **AND** commit successful work or roll back failed work when leaving the context

#### Scenario: Opening spatial connections

- **WHEN** client code enters a spatial database connection context
- **THEN** the system SHALL load SpatiaLite support for that connection

### Requirement: Scenarios are isolated

The system SHALL manage scenarios as registered project variants with their own scenario storage.

#### Scenario: Creating an empty scenario

- **WHEN** a new empty scenario is created
- **THEN** the system SHALL create scenario storage under `scenarios/<scenario_name>/`
- **AND** register the scenario in the root project database
- **AND** initialize scenario network tables without copying root network data

#### Scenario: Cloning a scenario

- **WHEN** a scenario is cloned
- **THEN** the system SHALL copy the active scenario's databases, matrices, and parameters where present
- **AND** register the new scenario in the root project database

### Requirement: Project closure releases resources

The system SHALL close project resources and clear active project context when a project is closed.

#### Scenario: Closing an open project

- **WHEN** `Project.close()` is called for an open project
- **THEN** the system SHALL commit pending project database work
- **AND** clean temporary project resources
- **AND** close and detach project log handlers
- **AND** deactivate the project context

