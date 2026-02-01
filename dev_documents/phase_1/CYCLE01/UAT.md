# User Acceptance Testing (UAT): Cycle 01

## 1. Test Scenarios

The goal of Cycle 01 is to establish a stable foundation. While there is no "scientific" output yet, the User Experience (UX) of configuration and setup is critical. A frustrated user who cannot get the system to start will never reach the physics results.

### Scenario 1.1: Project Initialisation
-   **ID**: UAT-C01-01
-   **Priority**: High
-   **Description**: A new user wants to start a project. They should be able to generate a template configuration file without copy-pasting from documentation.
-   **Success Criteria**: Running the `init` command creates a commented, valid `config.yaml` in the current directory.

### Scenario 1.2: Configuration Validation (The "Guard Rails")
-   **ID**: UAT-C01-02
-   **Priority**: Critical
-   **Description**: A user makes a typo in the config file (e.g., negative cutoff radius, missing element list). The system should catch this immediately upon startup, not 2 hours later during a simulation.
-   **Success Criteria**: The system prints a clear, human-readable error message pointing to the specific field that failed validation, and exits with a non-zero code.

### Scenario 1.3: Logging Verification
-   **ID**: UAT-C01-03
-   **Priority**: Medium
-   **Description**: The user wants to see what the system is doing.
-   **Success Criteria**:
    -   Console output is coloured and formatted (Info vs Error).
    -   A log file (`mlip.log`) is created and contains detailed debug information not shown on the console.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: System Configuration and Initialization

  Background:
    Given the mlip-auto CLI is installed

  Scenario: User initializes a new project
    When I run the command "mlip-auto init"
    Then a file named "config.yaml" should be created in the current directory
    And the file "config.yaml" should contain "project_name"
    And the file "config.yaml" should contain "potential"

  Scenario: User provides a valid configuration
    Given a file "config.yaml" exists with:
      """
      project_name: "TestProject"
      potential:
        elements: ["Ti", "O"]
        cutoff: 5.0
        seed: 42
      """
    When I run the command "mlip-auto check"
    Then the exit code should be 0
    And the output should contain "Configuration valid"

  Scenario: User provides an invalid configuration (Logical Error)
    Given a file "config.yaml" exists with:
      """
      project_name: "BadProject"
      potential:
        elements: ["Ti", "O"]
        cutoff: -1.0  # Invalid!
        seed: 42
      """
    When I run the command "mlip-auto check"
    Then the exit code should be 1
    And the output should contain "cutoff"
    And the output should contain "greater than 0"

  Scenario: System Logging
    Given a valid "config.yaml"
    When I run the command "mlip-auto check"
    Then a file named "mlip_pipeline.log" should be created
    And the log file should contain "Validation successful"
```
