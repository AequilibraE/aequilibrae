## ADDED Requirements

### Requirement: Migrations own schema evolution

The migration system SHALL own the behavior for moving existing project, transit, and results databases from earlier schemas to newer schemas.

#### Scenario: Planning schema evolution

- **WHEN** a change modifies tables, triggers, required fields, protected fields, or database metadata for existing projects
- **THEN** the change SHALL define the migration behavior needed to preserve or transform existing data

### Requirement: Migration specs do not duplicate the full base schema

The migration capability SHALL describe upgrade behavior rather than duplicating the complete current schema for newly created databases.

#### Scenario: Documenting a new table

- **WHEN** a new table is added to the base schema and existing projects must receive it
- **THEN** the base table contract SHALL be described by the owning data-model capability
- **AND** the upgrade path SHALL be described by the database migrations capability

