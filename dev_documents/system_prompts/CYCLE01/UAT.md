# Cycle 01 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1.1: Project Initialization (Happy Path)
*   **Priority:** High
*   **Description:** A user installs the package and runs the system with a valid configuration file. The system should start up, create the necessary workspace directories, and log the initialization status without errors.
*   **Input:** A valid `config.yaml` file.
*   **Expected Output:**
    *   Console output showing "PyAcemaker initialized".
    *   A new directory `workspace/` created.
    *   A log file `workspace/logs/system.log` containing startup messages.

### Scenario 1.2: Invalid Configuration Handling
*   **Priority:** Medium
*   **Description:** A user provides a configuration file with missing required fields (e.g., no DFT command specified) or invalid values (e.g., negative temperature). The system should gracefully exit with a clear error message pointing to the specific validation failure.
*   **Input:** `bad_config.yaml` (missing `dft` section).
*   **Expected Output:**
    *   System exit with non-zero code.
    *   Error message: "Configuration Error: Field 'dft' is required."

### Scenario 1.3: Workspace Resilience
*   **Priority:** Low
*   **Description:** The user runs the system, stops it, and runs it again. The system should detect the existing workspace and either resume (if configured) or warn the user, rather than crashing or blindly overwriting data.
*   **Input:** Run command twice.
*   **Expected Output:**
    *   Second run logs "Workspace already exists."

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Configuration Loading

  Scenario: User provides a valid configuration
    Given a file "config.yaml" exists with valid settings
    When I run "pyacemaker --config config.yaml"
    Then the exit code should be 0
    And a directory "workspace" should exist
    And the log file "workspace/logs/system.log" should contain "WorkflowManager initialized"

  Scenario: User provides an invalid configuration
    Given a file "bad_config.yaml" exists with "dft.command" missing
    When I run "pyacemaker --config bad_config.yaml"
    Then the exit code should be 1
    And the output should contain "validation error"
```
