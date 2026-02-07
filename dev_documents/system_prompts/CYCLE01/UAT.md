# Cycle 01 User Acceptance Testing (UAT)

## 1. Test Scenarios

### SCENARIO 01: Core CLI Execution
**Priority**: High
**Goal**: Verify the CLI can parse a configuration file and execute a full workflow using mock components.

**Steps**:
1.  Create a `config.yaml` with mock settings:
    ```yaml
    workdir: ./test_run
    max_cycles: 2
    generator:
      type: mock
    oracle:
      type: mock
    trainer:
      type: mock
    dynamics:
      type: mock
    ```
2.  Execute `mlip-pipeline run config.yaml`.
3.  Check output logs for "Cycle 1 complete", "Cycle 2 complete".
4.  Verify `test_run/` directory contains `potential_cycle_0.yace` and `potential_cycle_1.yace`.

## 2. Behavior Definitions

### Feature: Orchestrator Loop
**Scenario**: Active Learning Cycle Execution
  **Given** a valid configuration file specifying 2 cycles
  **And** all components are mocked
  **When** the user runs the `mlip-pipeline` command
  **Then** the system should initialize the Orchestrator
  **And** the Generator should produce initial structures
  **And** the Oracle should label them (dummy energy/forces)
  **And** the Trainer should create a dummy potential file
  **And** the Dynamics engine should simulate exploration and return "uncertain" structures
  **And** the loop should repeat exactly 2 times
  **And** the final exit code should be 0
