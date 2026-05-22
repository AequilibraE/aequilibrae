# Matrix IO Specification

## Purpose

This specification captures the current behavioral contract for AequilibraE matrix files, OMX interoperability, sparse matrices, and project matrix records.

## Requirements

### Requirement: Matrices can be memory-only or file-backed

The system SHALL support matrix objects that live only in memory and matrix objects backed by files on disk.

#### Scenario: Creating a memory matrix

- **WHEN** a matrix is created without a file path
- **THEN** the system SHALL allocate matrix data in memory
- **AND** allow computational views to be selected from its cores

#### Scenario: Creating a file-backed matrix

- **WHEN** a matrix is created with a supported file path
- **THEN** the system SHALL persist matrix data using the requested supported matrix format

### Requirement: Native matrix files are supported

The system SHALL support native AequilibraE `.aem` matrix files.

#### Scenario: Saving native matrix data

- **WHEN** a native matrix is saved
- **THEN** the system SHALL flush matrix data and metadata to the `.aem` file layout

#### Scenario: Loading native matrix data

- **WHEN** a native matrix file is loaded
- **THEN** the system SHALL expose its cores, indices, name, and description through the matrix object

### Requirement: OMX interoperability is supported

The system SHALL support reading and writing OMX matrices through the matrix API.

#### Scenario: Loading an OMX matrix

- **WHEN** an OMX file is loaded
- **THEN** the system SHALL expose available matrix cores and mappings

#### Scenario: Saving data to OMX

- **WHEN** matrix data is saved to an OMX file
- **THEN** the system SHALL write selected matrix cores and indices using OMX-compatible names and mappings

### Requirement: Computational views select matrix cores

The system SHALL require assignment and skimming procedures to operate on explicit matrix computational views.

#### Scenario: Selecting cores

- **WHEN** a computational view is selected
- **THEN** the system SHALL expose only the selected core data for computation

#### Scenario: Selecting unavailable cores

- **WHEN** a requested core is absent from the matrix
- **THEN** the system SHALL reject the computational view request

### Requirement: Project matrices are registered

The system SHALL maintain a registry of project matrix files in the project database.

#### Scenario: Registering an existing matrix file

- **WHEN** a matrix record is created for an existing `.omx` or `.aem` file
- **THEN** the system SHALL register the record in the project matrices table
- **AND** make the matrix available through the project matrices gateway

#### Scenario: Cleaning missing records

- **WHEN** a registered matrix file no longer exists on disk
- **THEN** the system SHALL be able to remove the stale registry record

### Requirement: Sparse matrix helpers interoperate

The system SHALL support sparse matrix conversion between AequilibraE sparse helpers, SciPy sparse matrices, and OMX datasets.

#### Scenario: Converting to SciPy

- **WHEN** a sparse matrix helper is converted to SciPy
- **THEN** the system SHALL return an equivalent SciPy sparse matrix

#### Scenario: Reading sparse data from OMX

- **WHEN** sparse data is loaded from an OMX file
- **THEN** the system SHALL expose it as either SciPy sparse data or AequilibraE sparse data according to the request

