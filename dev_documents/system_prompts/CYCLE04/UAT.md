# Cycle 04 UAT: Trainer (Pacemaker)

## 1. Test Scenarios

### Scenario 4.1: Dataset Creation and Update
*   **Goal:** Verify that ASE atoms can be converted to Pacemaker's training format.
*   **Steps:**
    1.  Create 10 valid ASE `Atoms` objects with energies and forces.
    2.  Use `Trainer.update_dataset()` to create `data/dataset_v1.pckl.gzip`.
    3.  Inspect the file (using `pandas.read_pickle` if needed).
*   **Expected Behavior:**
    *   The file exists.
    *   The dataframe inside has columns: `energy`, `force`, `atomic_numbers`, `positions`.

### Scenario 4.2: Training Execution
*   **Goal:** Verify that `pace_train` runs successfully on the dataset.
*   **Steps:**
    1.  Provide `data/dataset_v1.pckl.gzip` (from Scenario 4.1).
    2.  Run `Trainer.train(max_num_epochs=10)`.
    3.  Inspect the output directory.
*   **Expected Behavior:**
    *   `output_potential.yace` is created.
    *   `log.txt` shows "Training completed" (or similar success message).
    *   No segfaults or panics.

### Scenario 4.3: Active Set Filtering
*   **Goal:** Verify that `pace_activeset` selects a subset of structures.
*   **Steps:**
    1.  Create a large pool of 100 candidate structures (e.g., perturbed FCC lattices).
    2.  Run `ActiveSetSelector.select(pool, n_select=10)`.
    3.  Count the returned structures.
*   **Expected Behavior:**
    *   Exactly 10 structures are returned.
    *   The selected structures are distinct (check indices if possible).
    *   The orchestrator logs "Selected 10 active structures from pool of 100".

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Pacemaker Training

  Scenario: Convert ASE atoms to training dataset
    Given a list of 10 labeled ASE structures
    When I update the dataset
    Then a valid "dataset.pckl.gzip" file should be created
    And the file size should be non-zero

  Scenario: Execute training run
    Given a valid training dataset
    And a configuration with "max_num_epochs = 50"
    When I run the trainer
    Then a "potential.yace" file should be generated
    And the training log should report the final RMSE

  Scenario: Select active set
    Given a pool of 100 candidate structures
    When I request selection of 5 structures using D-Optimality
    Then the system should return exactly 5 structures
    And these structures should be the ones maximizing the information gain
```
