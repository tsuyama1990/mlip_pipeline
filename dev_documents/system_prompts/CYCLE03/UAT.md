# Cycle 03 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 3.1: Dataset Conversion
*   **Priority:** High
*   **Description:** The system must convert standard ASE Atoms objects (output from DFT) into the binary format required by Pacemaker without loss of precision in Forces or Energy.
*   **Input:** List of Atoms objects.
*   **Expected Output:** A `.pckl.gzip` file exists.

### Scenario 3.2: Training Execution
*   **Priority:** High
*   **Description:** Given a dataset and configuration, the system successfully invokes Pacemaker to train a potential.
*   **Input:** Valid dataset path, `TrainingConfig` (cutoff=5.0).
*   **Expected Output:**
    *   A file `potential.yace` is created.
    *   A log file showing training metrics (RMSE).

### Scenario 3.3: Active Set Selection
*   **Priority:** Medium
*   **Description:** When provided with a large dataset (e.g., 1000 structures), the system uses `pace_activeset` to reduce it to a smaller, information-rich dataset (e.g., 100 structures).
*   **Input:** Dataset with 1000 items, `active_set_size=100`.
*   **Expected Output:**
    *   A new dataset file containing exactly 100 items.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Potential Training

  Scenario: Train from Scratch
    Given a dataset file "data.pckl.gzip"
    And a training configuration with cutoff 5.0
    When I trigger the training phase
    Then the system should generate "input.yaml"
    And the system should run "pace_train"
    And a "potential.yace" file should be produced

  Scenario: Active Set Optimization
    Given a large dataset of 1000 structures
    When I request an active set of size 100
    Then the system should run "pace_activeset"
    And the resulting dataset should have 100 structures
```
