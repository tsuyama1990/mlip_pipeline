# CYCLE 01 UAT: Skeleton & Mock Loop

## 1. Test Scenarios

### Scenario 01.1: Hello World (Zero Config)
*   **Priority**: Critical
*   **Objective**: Verify that the system can be installed and run without crashing.
*   **Description**: The user installs the package and runs the help command.
*   **Success Criteria**:
    *   Command `mlip-pipeline --help` returns a formatted help message.

### Scenario 01.2: The Mock Loop
*   **Priority**: Critical
*   **Objective**: Verify the orchestration logic.
*   **Description**: The user provides a configuration file with `max_cycles: 3`. The system should output logs indicating 3 iterations of Generation -> Calculation -> Training.
*   **Input**: `config.yaml`
    ```yaml
    work_dir: "./workspace"
    max_cycles: 3
    random_seed: 42
    ```
*   **Success Criteria**:
    *   Process exits with code 0.
    *   Logs contain "Cycle 1/3 completed", "Cycle 2/3 completed", "Cycle 3/3 completed".
    *   Directory `./workspace` is created.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Core Orchestration

  Scenario: Running a basic loop with mock components
    GIVEN the package "mlip_autopipec" is installed
    AND a configuration file "config.yaml" with "max_cycles=3"
    WHEN I run the command "mlip-pipeline run config.yaml"
    THEN the exit code should be 0
    AND the output should contain "Starting Active Learning Cycle"
    AND the output should contain "MockExplorer generated structures"
    AND the output should contain "MockTrainer updated potential"
    AND the loop should iterate exactly 3 times
```
