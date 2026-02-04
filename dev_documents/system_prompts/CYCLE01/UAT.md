# CYCLE 01 UAT: Skeleton & Basic Loop

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-01-01** | High | CLI Smoke Test | Verify that the application is installed and the `help` command works. |
| **UAT-01-02** | High | Mock Loop Execution | Verify that the orchestrator runs through N cycles using Mock components and reports progress. |
| **UAT-01-03** | Medium | Config Validation | Verify that the system rejects a malformed `config.yaml`. |

## 2. Behavior Definitions

### Scenario: Mock Loop Execution
**GIVEN** a configuration file `config.yaml` with `execution_mode: mock`
**AND** `max_cycles` set to 3
**WHEN** the user runs `mlip-pipeline run config.yaml`
**THEN** the output should contain "Starting Cycle 1"
**AND** "MockExplorer generated candidates"
**AND** "MockOracle calculated properties"
**AND** "MockTrainer updated potential"
**AND** "Starting Cycle 2"
**AND** "Starting Cycle 3"
**AND** "Workflow completed successfully"

### Scenario: Config Validation Failure
**GIVEN** a configuration file `bad_config.yaml` missing required fields (e.g., `dft_settings`)
**WHEN** the user runs `mlip-pipeline run bad_config.yaml`
**THEN** the system should exit with an error code
**AND** print a clear validation error message identifying the missing field.
