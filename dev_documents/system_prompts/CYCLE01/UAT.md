# Cycle 01 UAT: Core Framework & Infrastructure

## 1. Test Scenarios

### Scenario 01: The "Hello World" of Configuration
**Priority**: High
**Description**: The user installs the package and tries to run it with a generated configuration file.
**Objective**: Verify CLI entry point and Config parsing.

**Steps**:
1.  Create a file `simple_config.yaml` with the following content:
    ```yaml
    project_name: "TestProject"
    dft:
      code: "qe"
      ecutwfc: 40.0
      kpoints: [2, 2, 2]
    training:
      code: "pacemaker"
      cutoff: 5.0
    ```
2.  Run the command: `python -m mlip_autopipec run simple_config.yaml`
3.  **Expected Result**: The program prints a formatted log message:
    `[INFO] PYACEMAKER initialized for project: TestProject`
    `[INFO] Configuration loaded successfully.`
    And exits with code 0.

### Scenario 02: The "Bad Config" Rejection
**Priority**: Medium
**Description**: The user provides an invalid configuration.
**Objective**: Verify Pydantic validation and Error reporting.

**Steps**:
1.  Create a file `bad_config.yaml` with missing required fields (e.g., missing `dft` section).
2.  Run the command: `python -m mlip_autopipec run bad_config.yaml`
3.  **Expected Result**: The program exits with a non-zero code and prints a helpful error message:
    `[ERROR] Configuration validation failed:`
    `Field 'dft' required`

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Application Initialization

  Background:
    Given the application is installed
    And the user has a terminal open

  Scenario: Loading a valid configuration
    Given a file named "valid_config.yaml" exists
    And the file contains valid YAML adhering to the schema
    When I run "python -m mlip_autopipec run valid_config.yaml"
    Then the exit code should be 0
    And the output should contain "Configuration loaded successfully"

  Scenario: Loading an invalid configuration
    Given a file named "invalid.yaml" exists with malformed data
    When I run "python -m mlip_autopipec run invalid.yaml"
    Then the exit code should be 1
    And the output should contain "Validation error"

  Scenario: Logging Infrastructure
    When I run the application with "--verbose" flag
    Then the log output should show timestamps
    And the log output should show the module name "mlip_autopipec.orchestration"
```
