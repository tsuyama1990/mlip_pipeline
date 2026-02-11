# Cycle 01 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario 1.1: Project Initialization
*   **Goal**: Verify that the CLI can initialize a project structure and create a default configuration file.
*   **Action**:
    1.  User runs `pyacemaker init my_project`.
    2.  User inspects the created directory.
*   **Success Criteria**:
    *   A directory `my_project/` is created.
    *   A `config.yaml` file exists inside with valid default values.
    *   A `data/` directory is created.

### Scenario 1.2: The Mock Loop (Core Workflow)
*   **Goal**: Verify that the Orchestrator can load the configuration, instantiate Mock components, and run a complete loop cycle without crashing.
*   **Action**:
    1.  User edits `config.yaml` to set `n_iterations: 2` and `mode: mock`.
    2.  User runs `pyacemaker run-loop`.
    3.  User inspects the output logs.
*   **Success Criteria**:
    *   The CLI prints "Cycle 1 Started", "Cycle 1 Completed", "Cycle 2 Started", "Cycle 2 Completed".
    *   The `workflow_state.json` file shows `current_iteration: 2`.
    *   No Python tracebacks or `KeyError` messages.

### Scenario 1.3: Configuration Validation (Error Handling)
*   **Goal**: Verify that invalid configurations are caught early with helpful error messages.
*   **Action**:
    1.  User intentionally breaks `config.yaml` (e.g., sets `n_iterations: -5` or deletes a required field).
    2.  User runs `pyacemaker run-loop`.
*   **Success Criteria**:
    *   The CLI prints a user-friendly error message: `Configuration Error: n_iterations must be positive`.
    *   The program exits with a non-zero status code.
    *   No raw Pydantic validation traceback is shown to the user (unless `--debug` is used).

## 2. Behavior Definitions (Gherkin Style)

### Feature: Project Initialization
**Scenario**: User initializes a new project
  **Given** the user is in an empty directory
  **When** the user runs `pyacemaker init test_project`
  **Then** a directory `test_project` should be created
  **And** the file `test_project/config.yaml` should exist
  **And** the file content should be valid YAML

### Feature: Workflow Execution
**Scenario**: User runs the mock loop
  **Given** a valid `config.yaml` with `mode: mock`
  **When** the user runs `pyacemaker run-loop`
  **Then** the orchestrator should run for the specified number of iterations
  **And** the state file `workflow_state.json` should be updated after each cycle
  **And** the log file should contain "Cycle Completed" messages

### Feature: Error Handling
**Scenario**: User provides invalid configuration
  **Given** a `config.yaml` with `n_iterations: -1`
  **When** the user runs `pyacemaker run-loop`
  **Then** the command should fail with exit code 1
  **And** the output should contain "validation error"
