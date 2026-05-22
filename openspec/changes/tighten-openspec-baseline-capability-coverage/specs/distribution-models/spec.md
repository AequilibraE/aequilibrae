## ADDED Requirements

### Requirement: Gravity application distributes demand

The system SHALL apply gravity models to produce origin-destination demand matrices from production and attraction vectors, impedance data, and configured deterrence functions.

#### Scenario: Applying a configured gravity model

- **WHEN** gravity application is executed with compatible productions, attractions, impedance, and model parameters
- **THEN** the system SHALL produce an origin-destination matrix with demand distributed according to the configured deterrence function
- **AND** preserve the configured numerical convergence controls

### Requirement: Gravity calibration estimates model parameters

The system SHALL calibrate supported gravity models against observed demand and impedance data.

#### Scenario: Calibrating a model

- **WHEN** calibration is run with observed matrix data and impedance values
- **THEN** the system SHALL estimate parameters for a supported gravity model function
- **AND** expose calibration convergence information through the distribution result interface

### Requirement: Synthetic gravity models persist configuration

The system SHALL represent synthetic gravity model configuration in a reusable object that can be saved and loaded.

#### Scenario: Loading a saved model

- **WHEN** a saved synthetic gravity model is loaded
- **THEN** the system SHALL restore its function type and parameter values for reuse in gravity application

### Requirement: IPF balances matrices to marginals

The system SHALL support iterative proportional fitting to balance an input matrix to target row and column marginals.

#### Scenario: Balancing with IPF

- **WHEN** IPF is executed with compatible seed matrix, row totals, and column totals
- **THEN** the system SHALL iteratively adjust the matrix until configured convergence criteria or iteration limits are reached
- **AND** expose convergence status for the run

### Requirement: Distribution workflows use configured defaults

The system SHALL use project parameters when an active project is available and package defaults otherwise.

#### Scenario: Running outside a project

- **WHEN** a distribution workflow is executed without an active project
- **THEN** the system SHALL fall back to package default parameters

