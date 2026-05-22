## ADDED Requirements

### Requirement: Base schema ownership is separate from migration ownership

The network data model SHALL own the current base tables, triggers, required fields, protected fields, and integrity constraints for newly created project databases.

#### Scenario: Planning a base schema change

- **WHEN** a change modifies the schema expected in newly created project databases
- **THEN** the change SHALL update the network data model requirements for the affected table, trigger, or field behavior
- **AND** use database migration requirements to describe how existing databases are upgraded

### Requirement: Network data model does not replace migrations

The network data model SHALL NOT be treated as sufficient documentation for upgrading existing project databases.

#### Scenario: Changing an existing database contract

- **WHEN** a change affects databases that may already exist in user projects
- **THEN** the change SHALL include migration behavior under the database migrations capability

