## ADDED Requirements

### Requirement: Project matrix files can be imported
The system SHALL provide a project matrix gateway endpoint that imports supported matrix files into the active project's matrix folder and registers them in the project matrix registry.

#### Scenario: Importing an external OMX file
- **WHEN** a user imports an existing external `.omx` matrix file through the project matrix gateway
- **THEN** the system SHALL copy the file into the project's `matrices` folder
- **AND** register a matrix record for the copied file
- **AND** make the imported matrix available through the project matrices gateway

#### Scenario: Importing an external native matrix file
- **WHEN** a user imports an existing external `.aem` matrix file through the project matrix gateway
- **THEN** the system SHALL copy the file into the project's `matrices` folder
- **AND** register a matrix record for the copied file
- **AND** make the imported matrix available through the project matrices gateway

#### Scenario: Rejecting unsupported matrix file types
- **WHEN** a user imports a file whose extension is not a supported project matrix format
- **THEN** the system SHALL reject the import before copying or registering the file

#### Scenario: Rejecting duplicate destination records
- **WHEN** the requested matrix record name or destination file name conflicts with an existing project matrix
- **THEN** the system SHALL reject the import before overwriting existing project matrix data
