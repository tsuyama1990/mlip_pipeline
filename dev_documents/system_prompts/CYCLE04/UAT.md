# Cycle 04 UAT: Trainer & Pacemaker Integration

## 1. Test Scenarios

### Scenario 4.1: Delta Learning Baseline
*   **ID**: S04-01
*   **Priority**: High
*   **Description**: Verify the system correctly subtracts the Lennard-Jones baseline from the dataset.
*   **Steps**:
    1.  Create a Dataset with 10 structures.
    2.  Set `baseline_type: lj`.
    3.  Call `compute_lj_baseline(dataset)`.
    4.  Inspect the resulting labels.
*   **Expected Result**:
    *   For structures where atoms overlap (distance < sigma), the `LJ` energy should be large positive.
    *   The `Residual` energy (Total - LJ) should be negative (typically binding).
    *   No NaNs or Infs.

### Scenario 4.2: Pacemaker Training (Mocked)
*   **ID**: S04-02
*   **Priority**: Critical
*   **Description**: Verify the Trainer generates the correct command and output file structure.
*   **Steps**:
    1.  Use `MockTrainer` configured to mimic `PacemakerTrainer` outputs.
    2.  Run `trainer.train(dataset)`.
*   **Expected Result**:
    *   A `.yace` file is created at the expected path.
    *   A `report.yaml` (dummy) is created.
    *   Logs show `pace_train --dataset ...` being executed.

### Scenario 4.3: Active Set Selection
*   **ID**: S04-03
*   **Priority**: Medium
*   **Description**: Verify that the Active Set algorithm reduces the dataset size.
*   **Steps**:
    1.  Create a dataset with 100 highly correlated structures (e.g., small perturbations of the same crystal).
    2.  Call `select_active_set(dataset, max_size=20)`.
*   **Expected Result**:
    *   The returned subset has <= 20 structures.
    *   The subset contains the most distinct structures (e.g., the original + large perturbations).

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Pacemaker Training

  Scenario: Generating training command
    GIVEN a training configuration with "cutoff=5.0" and "max_deg=8"
    WHEN I request the "pace_train" command
    THEN the command string should contain "--cutoff 5.0"
    AND the command string should contain "--max_deg 8"

  Scenario: Applying Delta Learning
    GIVEN a structure with two Argon atoms at 2.0 Angstrom distance
    AND a Lennard-Jones baseline (sigma=3.4, epsilon=0.01)
    WHEN I compute the baseline energy
    THEN the energy should be positive (repulsive)
    AND the Trainer should subtract this from the DFT energy
```
