# Cycle 04: Trainer & Active Learning - UAT

## 1. Test Scenarios

### Scenario 1: Basic ACE Training (Mock)
*   **ID**: UAT-04-001
*   **Objective**: Ensure the `PacemakerTrainer` can launch a training job.
*   **Pre-conditions**: Mock `pace_train` binary exists or is stubbed.
*   **Steps**:
    1.  Configure `TrainerConfig` with `max_epochs=10`.
    2.  Provide a dataset of 10 atoms.
    3.  Call `trainer.train()`.
*   **Expected Result**:
    *   A directory `output_potential` is created.
    *   A `.yace` file is generated.
    *   The `pace_train` command was executed with the correct arguments.

### Scenario 2: Active Set Selection
*   **ID**: UAT-04-002
*   **Objective**: Verify that the trainer uses `pace_activeset` to filter data.
*   **Pre-conditions**: Mock `pace_activeset` binary exists.
*   **Steps**:
    1.  Configure `TrainerConfig` with `active_set_size=5`.
    2.  Provide a dataset of 20 atoms.
    3.  Call `trainer.train()`.
*   **Expected Result**:
    *   The trainer first calls `pace_activeset` on the dataset.
    *   The output active set (5 atoms) is used as input for `pace_train`.
    *   The final potential is trained on only 5 structures.

### Scenario 3: Dataset Update
*   **ID**: UAT-04-003
*   **Objective**: Ensure new DFT data is correctly appended to the training set.
*   **Pre-conditions**: Existing dataset with 10 structures.
*   **Steps**:
    1.  Create 5 new structures (e.g., from Oracle).
    2.  Call `trainer.update_dataset(new_structures)`.
    3.  Inspect the dataset file.
*   **Expected Result**:
    *   The dataset file now contains 15 structures.
    *   The new structures have correct energies/forces.

## 2. Behavior Definitions

```gherkin
Feature: Automated Potential Training

  Scenario: Training from scratch
    Given I have a set of labeled structures
    And the Pacemaker configuration is valid
    When I invoke the trainer
    Then it should create a new potential file (.yace)
    And it should log the training error (RMSE)

  Scenario: Incremental Training (Active Learning)
    Given an existing potential and a new batch of data
    When I invoke the trainer with "initial_potential" set
    Then it should load the existing weights
    And it should perform fine-tuning (few epochs)
    And the new potential should have lower error on the new data

  Scenario: Active Set filtering
    Given a large dataset (1000 structures)
    And a target active set size of 100
    When I invoke the trainer
    Then it should run D-Optimality selection
    And it should train on the selected 100 structures only
```
