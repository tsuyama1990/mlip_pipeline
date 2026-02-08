# Cycle 01 UAT: Core Framework & Mocks

## 1. Test Scenarios

### Scenario 1.1: Environment Setup & Help Command
*   **ID**: S01-01
*   **Priority**: High
*   **Description**: Verify the CLI is installed and accessible.
*   **Steps**:
    1.  Install the package (`uv sync`, `pip install .`).
    2.  Run `mlip-pipeline --help`.
*   **Expected Result**: The help message is displayed, showing the usage and arguments.

### Scenario 1.2: Valid Config Execution (Mock)
*   **ID**: S01-02
*   **Priority**: Critical
*   **Description**: Run the full orchestration loop with a valid configuration file using mock components.
*   **Steps**:
    1.  Create `config_test.yaml` with `max_cycles: 2` and mock components selected.
    2.  Run `mlip-pipeline config_test.yaml`.
*   **Expected Result**:
    *   Command exits with code 0.
    *   Logs show entry into Cycle 1 and Cycle 2.
    *   Logs show "MockGenerator: Generated...", "MockOracle: Computed...", etc.

### Scenario 1.3: Invalid Config Rejection
*   **ID**: S01-03
*   **Priority**: Medium
*   **Description**: Verify that the system rejects invalid configuration files.
*   **Steps**:
    1.  Create `invalid.yaml` (missing `workdir`).
    2.  Run `mlip-pipeline invalid.yaml`.
*   **Expected Result**:
    *   Command exits with non-zero code.
    *   Error message explicitly mentions the missing field (Pydantic validation error).

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Core Orchestration

  Scenario: Running a basic loop with mocks
    GIVEN a configuration file "config.yaml" with:
      | field       | value      |
      | max_cycles  | 2          |
      | workdir     | ./work_01  |
      | generator   | mock       |
    WHEN I execute the command "mlip-pipeline config.yaml"
    THEN the exit code should be 0
    AND the log should contain "Cycle 1/2 started"
    AND the log should contain "Cycle 2/2 started"
    AND the directory "./work_01" should exist

  Scenario: Handling invalid configuration
    GIVEN a configuration file "bad_config.yaml" with missing "workdir"
    WHEN I execute the command "mlip-pipeline bad_config.yaml"
    THEN the exit code should be 1
    AND the output should contain "validation error"
```
