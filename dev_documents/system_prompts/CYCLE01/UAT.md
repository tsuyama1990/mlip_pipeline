# Cycle 01 UAT: System Initialization and Configuration

## 1. Test Scenarios

### Scenario 01: Successful System Initialization (Mock Mode)
-   **Priority**: Critical
-   **Description**: Verify that the system can read a valid configuration file and initialize the `MaceSurrogateOracle` (in Mock Mode) without errors.
-   **Execution**:
    1.  Create `tests/uat/configs/cycle01_valid.yaml`.
    2.  Run `python -m pyacemaker.main config.yaml`.
    3.  Check output for "System initialized successfully" and "MACE Oracle loaded (Mock)".

### Scenario 02: Invalid Configuration Handling
-   **Priority**: High
-   **Description**: Verify that the system gracefully rejects invalid configuration files (e.g., missing mandatory fields).
-   **Execution**:
    1.  Create `tests/uat/configs/cycle01_invalid.yaml` (omit `elements`).
    2.  Run `python -m pyacemaker.main config.yaml`.
    3.  Check output for `ValidationError` or clear error message.

## 2. Behavior Definitions (Gherkin)

### Feature: System Initialization

**Scenario: Load valid configuration and initialize components**
  GIVEN a valid `config.yaml` file with `mock_mode: true`
  WHEN I execute the `pyacemaker` command
  THEN the Orchestrator should parse the configuration successfully
  AND the `MaceSurrogateOracle` should be instantiated
  AND the system status should be "READY"

**Scenario: Reject invalid configuration**
  GIVEN an invalid `config.yaml` file (missing required fields)
  WHEN I execute the `pyacemaker` command
  THEN the Orchestrator should raise a `ConfigError` or `ValidationError`
  AND the system should exit with a non-zero status code
