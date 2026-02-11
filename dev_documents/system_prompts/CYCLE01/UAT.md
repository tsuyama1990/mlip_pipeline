# Cycle 01 User Acceptance Test (UAT)

## 1. Test Scenarios

### Scenario 1: Project Initialization & Configuration
**Priority**: P0 (Critical)
**Description**: Verify that the project structure is valid and the configuration system correctly parses valid inputs.
**Steps**:
1.  Create a sample `config.yaml` with valid settings.
2.  Run the CLI command: `mlip-runner run config.yaml`.
3.  Check the console output for success messages.
4.  Check if the working directory is created.

### Scenario 2: Invalid Configuration Handling
**Priority**: P1 (High)
**Description**: Verify that the system gracefully rejects invalid configurations with helpful error messages.
**Steps**:
1.  Create a sample `bad_config.yaml` with missing fields or invalid types (e.g., negative `max_iterations`).
2.  Run the CLI command: `mlip-runner run bad_config.yaml`.
3.  Check the console output for a clear `ValidationError` message (not a raw traceback).

## 2. Behavior Definitions (Gherkin)

### Feature: Configuration Loading

**Scenario**: Successful Initialization
    **Given** a valid `config.yaml` file with `max_iterations: 10`
    **When** the user runs `mlip-runner run config.yaml`
    **Then** the system should print "Starting Workflow..."
    **And** the directory `work_dir` should be created
    **And** the exit code should be 0

**Scenario**: Invalid Parameter
    **Given** an invalid `config.yaml` file with `max_iterations: -5`
    **When** the user runs `mlip-runner run config.yaml`
    **Then** the system should print "Validation Error: max_iterations must be positive"
    **And** the exit code should be 1
