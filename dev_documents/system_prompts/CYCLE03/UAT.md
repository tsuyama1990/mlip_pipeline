# Cycle 03 UAT: Machine Learning Core

## 1. Test Scenarios

### Scenario 03: "Learning from Data"
**Priority**: High
**Description**: Verify that the system can take a labeled dataset and produce a valid ACE potential file. This tests the `PacemakerWrapper` and `Dataset` management.

**Pre-conditions**:
-   `pacemaker` (or a mock script) is installed in the environment.
-   A `.pckl.gzip` file containing at least 10 labeled structures (e.g., from Cycle 02).

**Steps**:
1.  User creates a `config.yaml` with `trainer.type: pacemaker`.
2.  User runs `pyacemaker train --dataset data.pckl.gzip --config config.yaml` (New CLI command).

**Expected Outcome**:
-   Console shows progress bar (Epochs).
-   `potential.yace` is created in the output directory.
-   A `training_report.json` is created containing the final RMSE values.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Potential Training

  Scenario: Train ACE Potential from Dataset
    Given a dataset with 10 labeled structures
    And a Trainer configuration with "max_epochs: 1"
    When I request training
    Then the "pace_train" command should be executed
    And a "potential.yace" file should be generated
    And the training logs should report "RMSE Energy"
```
