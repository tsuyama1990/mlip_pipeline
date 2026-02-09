# Cycle 04 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

These tests verify the Trainer's ability to produce a valid potential file.

### Scenario 4.1: Data Conversion & Preparation
**Objective**: Ensure ASE structures are correctly formatted for Pacemaker.
**Priority**: High (P1)

*   **Setup**: A list of 10 `Structure` objects with random energies and forces.
*   **Action**: Call `utils.structures_to_pacemaker_dataframe`.
*   **Expected Outcome**:
    *   A `dataset.pckl.gzip` file is created.
    *   The file can be loaded by pandas.
    *   The DataFrame contains 'energy', 'forces', 'virial' columns.

### Scenario 4.2: Baseline Subtraction (Delta Learning)
**Objective**: Verify that the trainer correctly isolates the residual energy.
**Priority**: Critical (P0) - Physics correctness.

*   **Setup**:
    *   Structure S with DFT Energy $E_{DFT} = -100$.
    *   Baseline ZBL Energy $E_{ZBL} = +50$ (repulsive).
*   **Action**: Call `delta.subtract_baseline([S], type="ZBL")`.
*   **Expected Outcome**:
    *   The structure sent to training has target energy $E_{train} = -150$.
    *   The forces are similarly adjusted ($F_{train} = F_{DFT} - F_{ZBL}$).

### Scenario 4.3: Training Execution (Smoke Test)
**Objective**: Ensure the `pace_train` command is constructed and executed correctly.
**Priority**: High (P1) - Requires Pacemaker.

*   **Setup**: Config with `trainer.type="pacemaker"`. Small dataset.
*   **Action**: Call `train(dataset)`.
*   **Expected Outcome**:
    *   `input.yaml` is generated with correct settings (cutoff, basis size).
    *   `pace_train` runs without error.
    *   A `potential.yace` file is produced.
    *   The `Potential` object returned points to this file.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Potential Training

  Scenario: Preparing data for Delta Learning
    Given a set of structures labeled with DFT energies
    And a baseline potential of type "Lennard-Jones"
    When I prepare the dataset for training
    Then the baseline energy should be subtracted from the DFT energy
    And the baseline forces should be subtracted from the DFT forces
    And the resulting residual dataset should be saved to disk

  Scenario: Running Pacemaker
    Given a valid dataset "dataset.pckl.gzip"
    And a Trainer configuration specifying a cutoff of 5.0 Angstrom
    When I execute the training process
    Then a Pacemaker configuration file "input.yaml" should be generated
    And the external command "pace_train" should be invoked
    And a final potential file "output.yace" should be created
```
