# Cycle 03 UAT: Trainer & Potential Generation

## 1. Test Scenarios

### Scenario 01: Training Execution (Mocked)
**Priority**: Critical
**Description**: Verify that the `Trainer` correctly constructs the `pace_train` command and executes it (mocked) without error.
**Steps**:
1.  Create a python script `test_trainer.py`.
2.  Define `TrainerConfig` with `cutoff=5.0`, `order=3`.
3.  Mock `subprocess.run` to simulate `pace_train` success.
4.  Run `Trainer.train(dataset_path="data.pckl.gzip")`.
**Expected Result**:
-   Exit code 0.
-   `subprocess.run` called with `pace_train --cutoff 5.0 --order 3 ...`.
-   Returns path to `output_potential.yace`.

### Scenario 02: Active Set Selection (Mocked)
**Priority**: High
**Description**: Verify that `ActiveSetSelector` correctly constructs the `pace_activeset` command.
**Steps**:
1.  Create a python script `test_activeset.py`.
2.  Mock `subprocess.run` to simulate `pace_activeset` success.
3.  Run `ActiveSetSelector.select(candidates_path="pool.pckl.gzip", num_select=100)`.
**Expected Result**:
-   Exit code 0.
-   `subprocess.run` called with `pace_activeset --dataset pool.pckl.gzip --select 100 ...`.

### Scenario 03: Delta Learning Configuration
**Priority**: Medium
**Description**: Verify that if `delta_learning="zbl"` is set, the Trainer generates a ZBL potential file and passes it to Pacemaker as a baseline.
**Steps**:
1.  Create a python script `test_delta.py`.
2.  Define `TrainerConfig` with `delta_learning="zbl"`.
3.  Run `Trainer.train(...)`.
**Expected Result**:
-   A file `ZBL_baseline.yace` (or similar) is created in the output directory.
-   `pace_train` command includes `--baseline ZBL_baseline.yace`.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Trainer & Potential Generation

  Scenario: Train Potential
    Given a valid dataset path
    And a Trainer configuration
    When I request to train a potential
    Then the system should execute "pace_train" with correct arguments
    And the output should be a path to the trained potential file

  Scenario: Select Active Set
    Given a pool of candidate structures
    When I request to select 100 active structures
    Then the system should execute "pace_activeset"
    And the output should be a path to the selected dataset

  Scenario: Configure Delta Learning
    Given a Trainer configuration with "zbl" delta learning
    When I request to train a potential
    Then the system should generate a ZBL baseline potential
    And the training command should use this baseline
```
