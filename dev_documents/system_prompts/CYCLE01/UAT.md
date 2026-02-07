# Cycle 01 UAT: Core Infrastructure & Mocks

## 1. Test Scenarios

### SCENARIO 01: System Initialisation & Configuration
**Priority**: High
**Description**: Verify that the system can correctly parse a configuration file and initialise the necessary components.
**Pre-conditions**: A valid `config.yaml` file exists.
**Steps**:
1.  Run `mlip-pipeline run config.yaml`.
2.  Check the console output for initialisation messages.
**Expected Result**: The system logs the successful loading of the configuration and the instantiation of components (e.g., "Initialised MockOracle").

### SCENARIO 02: Mock Execution Loop
**Priority**: Medium
**Description**: Verify that the Mock components interact correctly through the defined interfaces.
**Pre-conditions**: A `config.yaml` specifying `type: mock` for all components.
**Steps**:
1.  Create a python script that manually instantiates the Mocks and calls their methods in sequence (Oracle -> Trainer -> Dynamics).
2.  Assert that data flows between them (e.g., Trainer receives data from Oracle).
**Expected Result**: The script completes without error, and artifacts (dummy potential files) are created.

## 2. Behaviour Definitions

```gherkin
Feature: Configuration Loading

  Scenario: Loading a valid configuration
    Given a file named "config_mock.yaml" with content:
      """
      workdir: "experiments/test_01"
      oracle:
        type: "mock"
        noise_level: 0.1
      trainer:
        type: "mock"
      dynamics:
        type: "mock"
      """
    When I run "mlip-pipeline run config_mock.yaml"
    Then the exit code should be 0
    And the log should contain "Configuration loaded successfully"
    And the log should contain "Initialised MockOracle"

  Scenario: Handling invalid configuration
    Given a file named "config_invalid.yaml" with content:
      """
      oracle:
        type: "unknown_type"
      """
    When I run "mlip-pipeline run config_invalid.yaml"
    Then the exit code should be 1
    And the log should contain "Validation Error"
```
