# Cycle 01: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: First-Time Setup (Hello World)
**Priority**: High
**Goal**: Verify that a new user can generate a configuration file and start the system without errors.
**Procedure**:
1.  Install the package.
2.  Run `mlip-auto init`.
3.  Check if `input.yaml` is created with sensible defaults.
4.  Run `mlip-auto run --config input.yaml` (or just `mlip-auto run`).
**Success Criteria**:
*   `input.yaml` exists.
*   System prints a "Welcome" message and logs the configuration.
*   System exits cleanly (since no phases are implemented yet) or starts the loop structure.

### Scenario 2: Invalid Configuration Handling
**Priority**: Medium
**Goal**: Verify that the system catches configuration errors before starting.
**Procedure**:
1.  Edit `input.yaml` and set a numeric field (e.g. `ecutwfc`) to a negative value.
2.  Run `mlip-auto validate --config input.yaml`.
**Success Criteria**:
*   System detects the error.
*   A clear error message is displayed: "Validation Error" (or similar Pydantic error).

## 2. Behavior Definitions

```gherkin
Feature: System Initialization

  Scenario: Generate default configuration
    GIVEN the system is installed
    WHEN the user runs "mlip-auto init"
    THEN a "input.yaml" file should be created in the current directory
    AND the file should contain default parameters for all phases

  Scenario: Start system with valid config
    GIVEN a valid "input.yaml" file
    WHEN the user runs "mlip-auto run"
    THEN the application should start the workflow
    AND the logger should output initialization messages
    AND the workflow state should be initialized to Cycle 0

  Scenario: Fail on invalid config
    GIVEN a "input.yaml" file with "ecutwfc = -5.0"
    WHEN the user runs "mlip-auto validate"
    THEN the application should exit with an error code
    AND the error message should mention validation failure
```
