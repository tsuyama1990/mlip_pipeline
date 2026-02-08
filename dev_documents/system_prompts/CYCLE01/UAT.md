# Cycle 01: Core & Mock UAT

## 1. Test Scenarios

### Scenario 01-01: CLI Help Command
**Priority**: High
**Goal**: Verify that the CLI application is correctly installed and accessible.
**Description**:
The user runs the main command without arguments or with `--help`.
**Expected Outcome**:
A clear help message showing available commands (e.g., `run`) and arguments.

### Scenario 01-02: Missing Configuration File
**Priority**: Medium
**Goal**: Verify error handling for missing input files.
**Description**:
The user attempts to run the system with a non-existent configuration file path.
**Expected Outcome**:
The application exits gracefully with a clear error message (e.g., "File not found: bad_config.yaml"), not a raw Python traceback.

### Scenario 01-03: Invalid Configuration Schema
**Priority**: High
**Goal**: Verify Pydantic validation.
**Description**:
The user provides a `config.yaml` with missing required fields (e.g., `workdir`) or invalid types (e.g., `max_cycles: "ten"`).
**Expected Outcome**:
The application exits with a structured validation error message indicating exactly which field is invalid.

### Scenario 01-04: The "Walking Skeleton" (Full Mock Run)
**Priority**: Critical
**Goal**: Verify the Orchestrator loop and Mock components integration.
**Description**:
1.  Create a `mock_config.yaml` specifying `mock` implementations for all components.
2.  Set `max_cycles: 2` and `workdir: ./test_output`.
3.  Run the CLI command: `python -m mlip_autopipec run mock_config.yaml`.
**Expected Outcome**:
-   The process completes successfully (exit code 0).
-   Logs show progression: "Starting Cycle 1", "MockGenerator generated 10 structures", "MockOracle computed forces", "MockTrainer trained potential", "Starting Cycle 2".
-   The directory `./test_output` is created and populated with artifacts (dummy potentials, logs).

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: CLI Entry Point

  Scenario: User requests help
    When I run "python -m mlip_autopipec --help"
    Then the exit code should be 0
    And the output should contain "Usage: python -m mlip_autopipec"

  Scenario: User provides invalid config path
    When I run "python -m mlip_autopipec run non_existent.yaml"
    Then the exit code should be 2 (or 1)
    And the output should contain "does not exist"

Feature: Orchestrator Loop with Mocks

  Scenario: Complete Active Learning Cycle with Mocks
    Given a configuration file "mock_config.yaml" with:
      | field        | value |
      | max_cycles   | 2     |
      | generator    | mock  |
      | oracle       | mock  |
      | trainer      | mock  |
      | dynamics     | mock  |
    When I run "python -m mlip_autopipec run mock_config.yaml"
    Then the exit code should be 0
    And the directory "test_output/cycle_01" should exist
    And the file "test_output/cycle_01/potential.yace" should exist
```
