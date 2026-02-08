# Cycle 01 UAT: Core Framework & Mocks

## 1. Test Scenarios

These scenarios verify the foundational architecture without needing any physics.

### Scenario 01-01: "The Hello World"
**Priority:** P0 (Critical)
**Description:** Verify that the system can read a configuration file, instantiate mock components, and run the active learning loop for the specified number of cycles.
**Success Criteria:**
-   The command `mlip-pipeline run config_mock.yaml` completes with exit code 0.
-   The console output clearly shows the lifecycle events:
    -   "Generator: Creating 5 structures..."
    -   "Oracle: Computing energies for 5 structures..."
    -   "Trainer: Training potential (Cycle 1)..."
    -   "Dynamics: Running MD simulation..."

### Scenario 01-02: "Configuration Validation"
**Priority:** P1 (High)
**Description:** Verify that the system correctly rejects invalid configuration files and provides helpful error messages.
**Success Criteria:**
-   **Missing Field:** If `workdir` is missing, the system prints "Field required: workdir".
-   **Invalid Type:** If `generator.type` is "quantum_magic" (unknown), the system prints "Unknown generator type: quantum_magic".
-   **Exit Code:** The process exits with a non-zero code (e.g., 1 or 2).

### Scenario 01-03: "CLI Help"
**Priority:** P2 (Medium)
**Description:** Verify that the CLI provides help documentation.
**Success Criteria:**
-   The command `mlip-pipeline --help` prints usage instructions.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Core Orchestrator Workflow

  Scenario: Successful Mock Execution
    Given a configuration file "config_mock.yaml" with:
      | field      | value  |
      | workdir    | ./runs |
      | max_cycles | 2      |
      | generator  | mock   |
      | oracle     | mock   |
      | trainer    | mock   |
      | dynamics   | mock   |
    When I run "mlip-pipeline run config_mock.yaml"
    Then the exit code should be 0
    And the output should contain "Starting Cycle 1"
    And the output should contain "Starting Cycle 2"
    And the output should contain "MockGenerator initialized"
    And the output should contain "MockOracle computed"

  Scenario: Invalid Configuration Handling
    Given a configuration file "bad_config.yaml" with:
      | field      | value  |
      | max_cycles | "two"  | # Invalid type (string instead of int)
    When I run "mlip-pipeline run bad_config.yaml"
    Then the exit code should be non-zero
    And the output should contain "validation error"
    And the output should contain "max_cycles"
```
