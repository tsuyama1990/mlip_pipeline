# Cycle 01 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Clean Start
**Priority**: High
**Goal**: Verify that the system initializes correctly from a fresh configuration.
**Procedure**:
1.  Create a `config_valid.yaml` with standard settings (mock paths).
2.  Run `python -m mlip_autopipec.main --config config_valid.yaml`.
**Expected Result**:
*   Command exits with code 0.
*   Directories `active_learning/`, `data/`, `potentials/` are created.
*   Log file shows "Orchestrator initialized".

### Scenario 2: Invalid Configuration
**Priority**: High
**Goal**: Verify "Fail-Fast" behavior.
**Procedure**:
1.  Create `config_invalid.yaml` (e.g., negative temperature, missing required fields).
2.  Run `python -m mlip_autopipec.main --config config_invalid.yaml`.
**Expected Result**:
*   Command exits with code 1.
*   Error message clearly indicates *which* field is invalid (Pydantic error message).

### Scenario 3: Resume Capability (State Persistence)
**Priority**: Medium
**Goal**: Verify that the system tracks its state.
**Procedure**:
1.  Run the system for 1 cycle (mocked).
2.  Interrupt or let it finish.
3.  Check `state.json` exists and contains `{"cycle": 1}`.
4.  Run the system again.
**Expected Result**:
*   Log says "Resuming from Cycle 1".

## 2. Behavior Definitions

```gherkin
Feature: System Initialization

  Scenario: User provides a valid configuration
    GIVEN a configuration file "config.yaml"
    AND the configuration specifies "iterations: 5"
    WHEN the "main.py" is executed
    THEN the "active_learning" directory should be created
    AND a log file should be generated
    AND the application should exit successfully

  Scenario: User provides an invalid configuration
    GIVEN a configuration file with "iterations: -1"
    WHEN the "main.py" is executed
    THEN the application should exit with an error
    AND the error message should mention "Input should be greater than 0"
```
