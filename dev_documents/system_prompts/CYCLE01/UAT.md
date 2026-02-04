# Cycle 01 UAT: The Skeleton Dry Run

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-01-01** | High | **Hello World Loop** | Verify that the system can run a full active learning loop using Mock components without crashing. |
| **UAT-01-02** | Medium | **Config Validation** | Verify that the system rejects invalid configuration files (e.g., negative temperature, missing required fields) with clear error messages. |

## 2. Behavior Definitions

### UAT-01-01: Hello World Loop

**GIVEN** a pristine environment with the `mlip-pipeline` package installed
**AND** a valid `config.yaml` specifying `execution_mode: mock` and `cycles: 3`
**WHEN** the user runs the command `mlip-pipeline run config.yaml`
**THEN** the system should output logs indicating the start of Cycle 1
**AND** the `MockExplorer` should report "Generated 10 candidate structures"
**AND** the `MockOracle` should report "Computed forces for 10 structures"
**AND** the `MockTrainer` should report "Potential trained: potential_001.yace"
**AND** this cycle should repeat 3 times
**AND** the process should exit with code 0

### UAT-01-02: Config Validation

**GIVEN** a configuration file `bad_config.yaml` where `training.cutoff` is set to `-5.0` (invalid)
**WHEN** the user runs `mlip-pipeline run bad_config.yaml`
**THEN** the system should exit with a non-zero error code
**AND** the error message should contain "Input should be greater than 0" (Pydantic validation error)
