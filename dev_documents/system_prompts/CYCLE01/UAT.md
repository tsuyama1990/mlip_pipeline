# Cycle 01: Core Framework & Orchestrator Skeleton - UAT

## 1. Test Scenarios

### Scenario 1: Verify Project Structure and Configuration
*   **ID**: UAT-01-001
*   **Objective**: Ensure the system can initialize and validate the configuration file.
*   **Pre-conditions**: Python 3.10+ installed.
*   **Steps**:
    1.  Create a `config.yaml` with valid settings for Mock components.
    2.  Run `python src/mlip_autopipec/main.py --config config.yaml`.
*   **Expected Result**: The system prints "Configuration Loaded Successfully" and initializes the components without error.

### Scenario 2: Verify Mock Component Loop
*   **ID**: UAT-01-002
*   **Objective**: Ensure the Orchestrator can execute a full cycle using Mock components.
*   **Pre-conditions**: Scenario 1 passes.
*   **Steps**:
    1.  Run the Orchestrator in Mock Mode.
    2.  Check the output directory.
*   **Expected Result**:
    *   Log file `pyacemaker.log` is created.
    *   A dummy potential file `potential_mock_001.yace` is created.
    *   The loop runs for at least 1 iteration and exits cleanly.

## 2. Behavior Definitions

```gherkin
Feature: Core Orchestrator Execution

  Scenario: User provides a valid configuration file
    Given a configuration file "config.yaml" exists
    And the file defines "mock" components for Generator, Oracle, Trainer, Dynamics
    When I run the command "python -m mlip_autopipec.main" with the config file
    Then the system should initialize the Orchestrator
    And the logger should record "Orchestrator initialized with Mock components"
    And the execution should complete with exit code 0

  Scenario: User provides an invalid configuration file
    Given a configuration file "invalid_config.yaml" exists with missing fields
    When I run the command "python -m mlip_autopipec.main" with the invalid config
    Then the system should print a "ValidationError" message
    And the execution should fail with exit code 1

  Scenario: Mock Active Learning Loop execution
    Given the Orchestrator is configured for Mock Mode
    When the active learning loop starts
    Then the MockGenerator should produce 10 structures
    And the MockOracle should label them with random energies
    And the MockTrainer should create a dummy potential file
    And the MockDynamics should return a "Halt" or "Complete" status
    And the Orchestrator should log "Cycle 1 complete"
```
