# Cycle 02 UAT: Data & Structure Generation

## 1. Test Scenarios

### Scenario 2.1: Dataset Handling
*   **ID**: S02-01
*   **Priority**: High
*   **Description**: Verify `Dataset` can save and load structures correctly.
*   **Steps**:
    1.  Create a script `test_dataset.py`.
    2.  Instantiate a `Dataset` object.
    3.  Create 10 dummy structures (different positions).
    4.  Call `dataset.append(structures)`.
    5.  Re-load the dataset from disk.
    6.  Assert the loaded structures match the originals.
*   **Expected Result**:
    *   File is created on disk (JSONL format).
    *   All 10 structures are recovered with exact numerical precision.

### Scenario 2.2: Structure Generation (Perturbation)
*   **ID**: S02-02
*   **Priority**: Medium
*   **Description**: Verify the Generator creates perturbed structures.
*   **Steps**:
    1.  Create a config specifying `generator_type: random_perturbation` and `sigma: 0.1`.
    2.  Run the pipeline.
    3.  Inspect the output dataset.
*   **Expected Result**:
    *   The generated structures have slightly different atomic positions than the input.
    *   The cell parameters remain the same (unless volume scaling is also enabled).

### Scenario 2.3: Supercell Creation
*   **ID**: S02-03
*   **Priority**: Medium
*   **Description**: Verify the Generator creates supercells.
*   **Steps**:
    1.  Create a config specifying `generator_type: supercell` and `size: [2, 2, 2]`.
    2.  Provide a primitive cell as input.
    3.  Run the pipeline.
*   **Expected Result**:
    *   The output structure contains 8x the number of atoms.
    *   The cell vectors are doubled in length.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Data Management

  Scenario: Appending structures to dataset
    GIVEN an empty dataset at "data/train.jsonl"
    WHEN I append a structure with 2 atoms
    THEN the file "data/train.jsonl" should exist
    AND the file size should be greater than 0
    AND I should be able to read 1 structure from the dataset

  Scenario: Generating perturbed structures
    GIVEN a base structure of FCC Aluminum
    AND a generator configured with "perturbation_sigma=0.1"
    WHEN the generator produces 5 candidates
    THEN each candidate should have the same number of atoms as the base
    BUT the positions should differ by approximately 0.1 Angstrom
```
