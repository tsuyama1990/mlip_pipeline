# Cycle 01 UAT: Core Infrastructure

## 1. Test Scenarios

| ID | Scenario Name | Description | Priority |
| :--- | :--- | :--- | :--- |
| **01-1** | **Project Initialization** | Verify that `mlip-runner init` creates a valid default configuration file. | High |
| **01-2** | **Mock Workflow Execution** | Verify that `mlip-runner run` executes a complete cycle using Mock components (no external binaries). | Critical |
| **01-3** | **Config Validation** | Verify that the system rejects invalid configuration files (e.g., negative temperatures, missing paths) with clear error messages. | Medium |

## 2. Behavior Definitions (Gherkin)

### Scenario 01-1: Project Initialization
```gherkin
GIVEN the user is in an empty directory
WHEN the user runs "mlip-runner init"
THEN a "config.yaml" file should be created
AND the file should contain valid default settings for "orchestrator", "generator", etc.
```

### Scenario 01-2: Mock Workflow Execution
```gherkin
GIVEN a valid "config.yaml" with "execution_mode: mock"
AND the configuration specifies "max_cycles: 2"
WHEN the user runs "mlip-runner run config.yaml"
THEN the system should log "Starting Cycle 1"
AND the system should log "Structure Generation (Mock) completed"
AND the system should log "DFT Calculation (Mock) completed"
AND the system should log "Training (Mock) completed"
AND the system should log "Validation (Mock) passed"
AND a directory "experiments/run_01" should be created
AND a file "experiments/run_01/potential.yace" should exist
```

### Scenario 01-3: Config Validation
```gherkin
GIVEN a "config.yaml" file where "orchestrator.max_cycles" is set to "-1"
WHEN the user runs "mlip-runner run config.yaml"
THEN the process should exit with an error
AND the error message should contain "Input should be greater than or equal to 1"
```
