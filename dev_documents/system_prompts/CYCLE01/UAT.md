# Cycle 01 UAT: The Hello World Run

## 1. Test Scenarios

### Scenario 1: System Initialization and Mock Execution
**Priority**: Critical
**Objective**: Verify that the system can read configuration, initialize components, and run the orchestration loop using mocks without crashing.

**Steps**:
1.  **Preparation**:
    *   Create a working directory `uat_cycle01`.
    *   Create a `config.yaml` with the following content:
        ```yaml
        work_dir: "./uat_cycle01/run"
        max_cycles: 3
        oracle:
          type: "mock"
        trainer:
          type: "mock"
        explorer:
          type: "mock"
        ```
2.  **Execution**:
    *   Run the command: `python -m mlip_autopipec.main config.yaml`
3.  **Verification**:
    *   Check console output for "Orchestrator started".
    *   Check that the directory `uat_cycle01/run` was created.
    *   Check for the existence of `uat_cycle01/run/potentials/generation_001.yace` (dummy file).
    *   Verify the process exits with code 0.

## 2. Behavior Definitions

```gherkin
Feature: Core Orchestration

  Scenario: Running a full mock cycle
    GIVEN a valid configuration file "config.yaml"
    AND the configuration specifies "type: mock" for all components
    WHEN the system is executed with "python -m mlip_autopipec.main config.yaml"
    THEN the system should initialize without errors
    AND the "Orchestrator" should run for the defined "max_cycles"
    AND the "MockTrainer" should produce a dummy potential file in each cycle
    AND the "MockOracle" should log that it computed energies
    AND the application should exit successfully
```
