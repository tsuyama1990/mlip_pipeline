# Cycle 01: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: First-Time Setup (Hello World)
**Priority**: High
**Goal**: Verify that a new user can generate a configuration file and start the system without errors.
**Procedure**:
1.  Install the package.
2.  Run `mlip-auto init`.
3.  Check if `config.yaml` is created with sensible defaults.
4.  Run `mlip-auto run --config config.yaml`.
**Success Criteria**:
*   `config.yaml` exists.
*   System prints a "Welcome" message and logs the configuration.
*   System exits cleanly (since no phases are implemented yet).

### Scenario 2: Invalid Configuration Handling
**Priority**: Medium
**Goal**: Verify that the system catches configuration errors before starting.
**Procedure**:
1.  Edit `config.yaml` and set `temperature` to `-100`.
2.  Run `mlip-auto run --config config.yaml`.
**Success Criteria**:
*   System does **not** start.
*   A clear error message is displayed: "Validation Error: Temperature must be positive." (or similar Pydantic error).

## 2. Behavior Definitions

```gherkin
Feature: System Initialization

  Scenario: Generate default configuration
    GIVEN the system is installed
    WHEN the user runs "mlip-auto init"
    THEN a "config.yaml" file should be created in the current directory
    AND the file should contain default parameters for all phases

  Scenario: Start system with valid config
    GIVEN a valid "config.yaml" file
    WHEN the user runs "mlip-auto run"
    THEN the application should start
    AND the logger should output "PyAcemaker initialized"
    AND the workflow state should be initialized to Cycle 0

  Scenario: Fail on invalid config
    GIVEN a "config.yaml" file with "cutoff = -5.0"
    WHEN the user runs "mlip-auto run"
    THEN the application should exit with an error code
    AND the error message should mention "cutoff must be positive"
```
