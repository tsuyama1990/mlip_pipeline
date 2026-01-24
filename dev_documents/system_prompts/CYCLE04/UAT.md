# Cycle 04 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 04-01: Dataset Preparation
- **Priority**: High
- **Description**: Convert ASE atoms with calculator results to training file.
- **Steps**:
  1. Create 10 atoms with random energies/forces.
  2. Run `DatasetBuilder`.
- **Expected Result**: `train.xyz` is created. `ase.io.read('train.xyz')` loads the properties correctly.

### Scenario 04-02: Training Execution
- **Priority**: Critical
- **Description**: End-to-end wrapper execution.
- **Steps**:
  1. Setup `PacemakerWrapper`.
  2. Point to `train.xyz`.
  3. Call `train()`.
- **Expected Result**: (With mock) The function returns a path to `potential.yace`.

### Scenario 04-03: Active Set Selection
- **Priority**: Medium
- **Description**: Filter structures based on D-optimality.
- **Steps**:
  1. Provide 100 candidate structures.
  2. Call `select_active_set(n=10)`.
- **Expected Result**: Returns exactly 10 indices.

## 2. Behavior Definitions

```gherkin
Feature: Training Orchestrator

  Scenario: Generate Pacemaker Config
    GIVEN a TrainingConfig with cutoff=5.0
    WHEN the wrapper generates input.yaml
    THEN the file should contain "cutoff: 5.0"
    AND the file should specify the dataset path

  Scenario: Parse Training Logs
    GIVEN a log file from a finished run
    WHEN the metrics parser reads it
    THEN it should report the final RMSE for Energy and Forces
```
