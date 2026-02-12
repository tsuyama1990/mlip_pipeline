# Cycle 01 UAT: Core Infrastructure

## 1. Test Scenarios

### Scenario 1: System Initialization
*   **ID**: S01-01
*   **Goal**: Verify that the CLI tool is installed and can initialize a new project.
*   **Priority**: Critical.
*   **Steps**:
    1.  Install the package: `uv sync`.
    2.  Run `mlip-runner init`.
    3.  Check that `config.yaml` is created with default values.
    4.  Check that `work_dir` is not created yet.

### Scenario 2: Configuration Validation
*   **ID**: S01-02
*   **Goal**: Verify that invalid configurations are rejected.
*   **Priority**: High.
*   **Steps**:
    1.  Modify `config.yaml` to have invalid types (e.g., `max_iterations: "ten"`).
    2.  Run `mlip-runner run config.yaml`.
    3.  Assert that the program exits with a clear Pydantic validation error message.

### Scenario 3: State Persistence
*   **ID**: S01-03
*   **Goal**: Verify that workflow state is saved and can be resumed (mock).
*   **Priority**: Medium.
*   **Steps**:
    1.  Run a dummy workflow that saves state `iteration=1`.
    2.  Interrupt execution (Ctrl+C).
    3.  Restart.
    4.  Assert that it resumes from `iteration=1` (logs "Resuming from iteration 1").

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: CLI Entry Point

  Scenario: User initializes a new project
    Given the package "mlip-pipeline" is installed
    When I run the command "mlip-runner init"
    Then a file named "config.yaml" should be created in the current directory
    And the file should contain "orchestrator:" section

  Scenario: User runs with invalid config
    Given a file named "bad_config.yaml" exists
    And "max_iterations" is set to "invalid_string"
    When I run "mlip-runner run bad_config.yaml"
    Then the process should exit with status code 1
    And the output should contain "validation error"

Feature: State Management

  Scenario: Atomic State Saving
    Given a workflow is running
    When the state is saved to "workflow_state.json"
    Then the file should contain valid JSON
    And "iteration" should be an integer
```
