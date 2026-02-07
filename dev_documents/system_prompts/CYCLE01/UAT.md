# Cycle 01 UAT: The Hello World of Atoms

## 1. Test Scenarios

### 1.1. Scenario: The "Hello World" Run (Mock)
**ID**: UAT-CY01-001
**Priority**: High
**Description**: The user executes the `main.py` CLI with a basic configuration file. The goal is to confirm that the entire orchestration loop (from initialization to completion) runs without crashing, using the Mock implementations of all components.

**Steps:**
1.  **Setup**: Create a file named `config.yaml` with the following content:
    ```yaml
    project_name: "test_project"
    orchestrator:
      max_iterations: 5
    oracle:
      type: "mock"
    trainer:
      type: "mock"
    dynamics:
      type: "mock"
    ```
2.  **Execution**: Run the command:
    ```bash
    python -m mlip_autopipec.main run config.yaml
    ```
3.  **Observation**: The terminal should display logs indicating the progression of the active learning loop (Iteration 1 -> Exploration -> Labelling -> Training -> Validation).
4.  **Verification**: A directory `active_learning/iter_001/` should be created. Inside, dummy potential files (e.g., `potential.yace`) should exist. The command should exit with code 0.

### 1.2. Scenario: Configuration Validation (Error Handling)
**ID**: UAT-CY01-002
**Priority**: Medium
**Description**: The user provides a malformed configuration file. The system should gracefully catch the error and provide a helpful message, rather than crashing with a raw Python traceback.

**Steps:**
1.  **Setup**: Create `bad_config.yaml`:
    ```yaml
    project_name: "bad_project"
    oracle:
      type: "unknown_type" # Invalid type
    ```
2.  **Execution**: Run `python -m mlip_autopipec.main run bad_config.yaml`.
3.  **Observation**: The output should contain a clear error message like `ValidationError: Input should be 'mock' or 'quantum_espresso'`.
4.  **Verification**: The program exits with a non-zero status code.

## 2. Behaviour Definitions (Gherkin)

### Feature: Active Learning Orchestration

**Scenario**: Successful execution of a Mock Active Learning Loop

**GIVEN** a valid configuration file `config.yaml` specifying `type: mock` for all components
**AND** the `max_iterations` is set to 3
**WHEN** the user executes the command `python -m mlip_autopipec.main run config.yaml`
**THEN** the system should initialize the Orchestrator
**AND** the system should run for exactly 3 iterations (or until convergence is simulated)
**AND** the log output should show "Starting Iteration 1", "Starting Iteration 2", etc.
**AND** the final log message should be "Active Learning Cycle Completed Successfully"
**AND** the directory `active_learning/iter_003` should exist
**AND** the process should exit with code 0

**Scenario**: Invalid Configuration Handling

**GIVEN** a configuration file `bad_config.yaml` with a missing required field `project_name`
**WHEN** the user executes the command `python -m mlip_autopipec.main run bad_config.yaml`
**THEN** the system should print a validation error message to stderr
**AND** the error message should mention "Field required" and "project_name"
**AND** the system should NOT start the orchestration loop
**AND** the process should exit with a non-zero code
