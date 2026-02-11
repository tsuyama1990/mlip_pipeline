# Cycle 01 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1.1: CLI Health Check
*   **Priority**: Critical
*   **Goal**: Verify the application is installed and reachable.
*   **Action**: Run `mlip-runner --help`.
*   **Expectation**: Typer help message is displayed, showing the `run` command.

### Scenario 1.2: First Run (Clean State)
*   **Priority**: High
*   **Goal**: Verify the system can initialize a new project.
*   **Action**:
    1.  Create a `config.yaml` with:
        ```yaml
        orchestrator:
          work_dir: "test_run_01"
          max_cycles: 2
        ```
    2.  Run `mlip-runner run config.yaml`.
*   **Expectation**:
    *   Directory `test_run_01` is created.
    *   File `test_run_01/mlip.log` exists and contains "Starting cycle 1...".
    *   File `test_run_01/workflow_state.json` exists and shows `iteration: 2` (or final state).
    *   Console output shows progress.

### Scenario 1.3: State Resumption (Mock Interrupt)
*   **Priority**: Medium
*   **Goal**: Verify the system can pick up where it left off.
*   **Action**:
    1.  Manually edit `test_run_01/workflow_state.json` to set `iteration: 1`.
    2.  Run `mlip-runner run config.yaml`.
*   **Expectation**:
    *   Log file shows "Resuming from iteration 1...".
    *   Orchestrator runs cycle 1 and 2, then finishes.

### Scenario 1.4: Invalid Configuration
*   **Priority**: Low
*   **Goal**: Verify error handling.
*   **Action**: Run with a config missing `work_dir`.
*   **Expectation**: The program exits with a clear validation error (Pydantic traceback or friendly message), not a generic crash.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Workflow Orchestration

  Scenario: User starts a new simulation
    Given a configuration file "config.yaml"
    And the working directory "sim_01" does not exist
    When I run "mlip-runner run config.yaml"
    Then the directory "sim_01" should be created
    And the log file "sim_01/mlip.log" should contain "Initialization complete"
    And the state file "sim_01/workflow_state.json" should exist

  Scenario: User resumes a simulation
    Given a working directory "sim_01" with state iteration=1
    When I run "mlip-runner run config.yaml"
    Then the system should log "Resuming from iteration 1"
    And the final state iteration should be equal to max_cycles
```
