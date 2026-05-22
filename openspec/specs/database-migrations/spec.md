# Database Migrations Specification

## Purpose

This specification captures the current behavioral contract for schema creation, migration discovery, migration execution, and database transaction handling.

## Requirements

### Requirement: Migrations are listed explicitly

The system SHALL discover migrations from the package migration listing files.

#### Scenario: Loading migrations

- **WHEN** a migration manager is created for a migration listing file
- **THEN** the system SHALL load each listed SQL or Python migration path
- **AND** sort migrations by numeric migration ID

#### Scenario: Rejecting invalid migrations

- **WHEN** a migration has a missing file, negative ID, duplicate ID, or unsupported extension
- **THEN** the system SHALL reject the migration set

### Requirement: Migration status is tracked

The system SHALL track each migration's status in the target database.

#### Scenario: Marking migrations as seen

- **WHEN** migrations are marked as seen
- **THEN** missing migration records SHALL be inserted with `MISSING` status

#### Scenario: Creating a new database

- **WHEN** a new database is created from the current schema
- **THEN** the system SHALL mark current migrations as seen and skipped because the schema already includes their effects

### Requirement: Applicable migrations are ordered

The system SHALL apply migrations in ID order and detect out-of-order migration states.

#### Scenario: Finding applicable migrations

- **WHEN** migrations are checked for an existing database
- **THEN** the system SHALL identify missing migrations after the last applied or skipped migration

#### Scenario: Detecting out-of-order state

- **WHEN** an applied migration appears after a missing migration
- **THEN** the system SHALL report an out-of-order state that requires manual intervention

### Requirement: SQL and Python migrations are supported

The system SHALL support migrations implemented as SQL scripts or Python functions.

#### Scenario: Applying SQL migration

- **WHEN** an SQL migration is applied
- **THEN** the system SHALL execute the SQL script against the main migration connection
- **AND** mark the migration as applied after successful execution

#### Scenario: Applying Python migration

- **WHEN** a Python migration is applied
- **THEN** the system SHALL import the migration module
- **AND** call its `migrate` function with the provided database connections
- **AND** mark the migration as applied after successful execution

### Requirement: Multi-database migrations are transactional

The system SHALL preserve transaction integrity when migrations span multiple databases.

#### Scenario: Applying migration with multiple connections

- **WHEN** a migration is applied with project, transit, or results database connections
- **THEN** the system SHALL enter manual transactions for all provided connections
- **AND** commit all successful work or roll back failed work as one migration operation

### Requirement: Project upgrade applies migrations

The system SHALL upgrade project databases by applying applicable network and transit migrations.

#### Scenario: Upgrading a project

- **WHEN** `Project.upgrade()` is called
- **THEN** the system SHALL apply applicable project database migrations
- **AND** apply transit database migrations when a transit database exists and is not ignored

#### Scenario: Ignoring a database

- **WHEN** a project upgrade is called with an ignore flag
- **THEN** the system SHALL skip that database
- **AND** warn that ignoring migrations can leave the project in an incompatible state

