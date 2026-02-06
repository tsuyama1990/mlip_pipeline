# Cycle 01 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario 1.1: Environment Setup & Help Command
**Priority**: Critical
**Description**: Verify that the application is correctly installed and the CLI entry point is accessible.
**Steps**:
1.  Install the package (`uv pip install -e .`).
2.  Run the command `mlip-pipeline --help`.
3.  Run the command `mlip-pipeline run --help`.

### Scenario 1.2: Configuration Validation
**Priority**: High
**Description**: Verify that the system correctly rejects invalid configurations and accepts valid ones.
**Steps**:
1.  Create a file `bad_config.yaml` with missing fields.
2.  Run `mlip-pipeline run --config bad_config.yaml`.
3.  Expect an error message explaining the missing field.
4.  Create a file `good_config.yaml` with all required fields.
5.  Run `mlip-pipeline run --config good_config.yaml`.
6.  Expect success.

### Scenario 1.3: "Mock Mode" Execution
**Priority**: High
**Description**: Verify the internal data flow of the Orchestrator using Mock components.
**Steps**:
1.  Set the configuration to use `mock` mode (implied for Cycle 01).
2.  Run the pipeline.
3.  Check the logs to see the sequence:
    *   "Starting Cycle 1..."
    *   "Explorer produced X candidates."
    *   "Oracle labeled X structures."
    *   "Trainer produced potential version 1."
    *   "Cycle 1 completed."

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: CLI Entry Point

  Scenario: User asks for help
    Given the package is installed
    When I run "mlip-pipeline --help"
    Then the exit status should be 0
    And the output should contain "Usage: mlip-pipeline"

  Scenario: User provides invalid configuration
    Given a configuration file "invalid.yaml" missing "max_cycles"
    When I run "mlip-pipeline run --config invalid.yaml"
    Then the exit status should be 1
    And the output should contain "validation error"

  Scenario: Running the Mock Loop
    Given a valid configuration file "config.yaml"
    When I run "mlip-pipeline run --config config.yaml"
    Then the exit status should be 0
    And the log should contain "Orchestrator initialization complete"
    And the log should contain "Cycle 01 finished"
```
