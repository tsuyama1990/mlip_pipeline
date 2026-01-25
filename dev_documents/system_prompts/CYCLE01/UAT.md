# Cycle 01 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Successful System Initialization
*   **ID**: UAT-01-01
*   **Priority**: Critical
*   **Description**: The user provides a valid configuration file. The system should read it, validate the parameters, create the necessary project directories, and log a success message.
*   **Success Criteria**:
    *   Command exits with code 0.
    *   `work_dir` specified in config is created.
    *   `system.log` contains "Orchestrator initialized successfully".

### Scenario 2: Invalid Configuration Handling
*   **ID**: UAT-01-02
*   **Priority**: High
*   **Description**: The user provides a configuration file with missing required fields (e.g., no `qe_command`) or invalid values (e.g., negative `temperature`).
*   **Success Criteria**:
    *   Command exits with a non-zero error code.
    *   Console output displays a clear, human-readable error message indicating exactly which field failed validation.
    *   System does *not* crash with a raw Python stack trace.

### Scenario 3: Logging Verification
*   **ID**: UAT-01-03
*   **Priority**: Medium
*   **Description**: Verify that the logging system correctly separates console and file output.
*   **Success Criteria**:
    *   Console shows high-level info (e.g., "Starting PyAcemaker...").
    *   Log file contains detailed debug info (e.g., "Loading config from path/to/config.yaml").

## 2. Behavior Definitions

```gherkin
Feature: System Initialization

  As a researcher
  I want to initialize the system with a single configuration file
  So that I can start the MLIP generation process without writing code

  Scenario: Valid Configuration
    GIVEN a configuration file "config.yaml" with valid paths and parameters
    WHEN I run "python -m mlip_autopipec.app --config config.yaml"
    THEN the exit code should be 0
    AND a directory "experiments/run_001" should be created
    AND the log file "experiments/run_001/system.log" should exist

  Scenario: Invalid DFT Configuration
    GIVEN a configuration file "bad_config.yaml" where "dft.kspacing" is -0.1
    WHEN I run "python -m mlip_autopipec.app --config bad_config.yaml"
    THEN the exit code should be 1
    AND the output should contain "Input should be greater than 0"
```
