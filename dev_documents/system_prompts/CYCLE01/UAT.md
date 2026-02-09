# Cycle 01 UAT: Core Framework

## 1. Test Scenarios

### Scenario 1.1: Happy Path Configuration
*   **Goal:** Verify that a valid configuration file loads successfully and instantiates all components.
*   **Steps:**
    1.  Create a `config.yaml` with valid settings for all modules (using "MOCK" types where available).
    2.  Run `python src/mlip_autopipec/main.py --config config.yaml`.
*   **Expected Behavior:**
    *   Console output: "Orchestrator initialized successfully".
    *   Log file: "DEBUG: Config loaded from config.yaml".
    *   Log file: "INFO: StructureGenerator initialized".
    *   Log file: "INFO: Oracle initialized".
    *   Log file: "INFO: Pipeline initialized".

### Scenario 1.2: Invalid Configuration Handling
*   **Goal:** Verify that the system rejects invalid configurations with clear error messages.
*   **Steps:**
    1.  Create a `bad_config.yaml` with `temperature: -100` (physically impossible) or `oracle: { type: "INVALID_TYPE" }`.
    2.  Run `python src/mlip_autopipec/main.py --config bad_config.yaml`.
*   **Expected Behavior:**
    *   Process exits with non-zero code.
    *   Console output: "ValidationError: 1 validation error for Config... temperature must be >= 0".

### Scenario 1.3: Logging Verification
*   **Goal:** Ensure logs are written correctly for debugging.
*   **Steps:**
    1.  Run the Happy Path scenario.
    2.  Inspect the generated `logs/pyacemaker.log`.
*   **Expected Behavior:**
    *   File exists.
    *   Contains timestamps and log levels.
    *   Contains "Orchestrator created".

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: System Initialization

  Scenario: Load valid configuration
    Given a valid configuration file "config.yaml"
    When I run the orchestrator with this config
    Then the system should initialize without errors
    And the log file should contain "Orchestrator initialized"
    And the factory should instantiate a "MockGenerator" component

  Scenario: Reject invalid configuration
    Given a configuration file with "temperature = -50"
    When I run the orchestrator
    Then the system should exit with an error
    And the error message should mention "validation error"

  Scenario: Ensure clean exit on missing file
    Given a non-existent configuration file "ghost.yaml"
    When I run the orchestrator
    Then the system should exit with an error
    And the error message should say "File not found"
```
