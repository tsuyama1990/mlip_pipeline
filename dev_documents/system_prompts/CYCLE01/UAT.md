# Cycle 01 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1.1: Project Initialization
*   **ID**: UAT-01-01
*   **Priority**: High
*   **Description**: A new user wants to start a project. They should be able to generate a default configuration file using the CLI.
*   **Steps**:
    1.  Open terminal.
    2.  Run `mlip-auto init`.
    3.  Check if `config.yaml` exists in the current directory.
    4.  Inspect `config.yaml` content.

### Scenario 1.2: Configuration Validation
*   **ID**: UAT-01-02
*   **Priority**: Medium
*   **Description**: The system should reject invalid configuration files.
*   **Steps**:
    1.  Run `mlip-auto init`.
    2.  Edit `config.yaml` and remove a required field (e.g., project name).
    3.  Run `mlip-auto run-loop`.
    4.  Verify that the system exits with a helpful error message.

### Scenario 1.3: Loop Startup and State Persistence
*   **ID**: UAT-01-03
*   **Priority**: High
*   **Description**: The user runs the loop, and the system initializes the workflow state.
*   **Steps**:
    1.  Run `mlip-auto init`.
    2.  Run `mlip-auto run-loop`.
    3.  Verify console logs show "Workflow initialized".
    4.  Check if `workflow_state.json` (or similar) is created.

## 2. Behavior Definitions

```gherkin
Feature: CLI and Configuration

  Scenario: User initializes a new project
    GIVEN I am in an empty directory
    WHEN I run "mlip-auto init"
    THEN a file named "config.yaml" should be created
    AND "config.yaml" should contain default settings

  Scenario: User runs the loop with valid config
    GIVEN I have a valid "config.yaml"
    WHEN I run "mlip-auto run-loop"
    THEN the exit code should be 0
    AND I should see "Starting MLIP Active Learning Loop" in the output
    AND a "workflow_state.json" file should be created

  Scenario: User runs the loop with invalid config
    GIVEN I have a "config.yaml" with missing required fields
    WHEN I run "mlip-auto run-loop"
    THEN the exit code should be non-zero
    AND I should see "Validation Error" in the output
```
