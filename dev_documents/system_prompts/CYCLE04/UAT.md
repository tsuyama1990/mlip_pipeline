# Cycle 04 UAT: Trainer (Pacemaker Integration)

## 1. Test Scenarios

### Scenario 1: Dataset Management
*   **ID**: S04-01
*   **Goal**: Verify that atomistic data can be saved and loaded efficiently.
*   **Priority**: Critical.
*   **Steps**:
    1.  Create 100 `ase.Atoms` objects with random positions and energies.
    2.  Save to `test_dataset.pckl.gzip`.
    3.  Read back.
    4.  Assert `len(dataset) == 100`.
    5.  Assert `dataset[0].energy` is correct.

### Scenario 2: Active Set Selection (D-Optimality)
*   **ID**: S04-02
*   **Goal**: Verify that `pace_activeset` selects distinct structures.
*   **Priority**: High.
*   **Steps**:
    1.  Create 10 highly similar structures (perturbed) + 1 distinct one.
    2.  Run `ActiveSetSelector.select(candidates, n=2)`.
    3.  Assert that the distinct structure is selected.
    4.  Assert that only 2 structures are returned.

### Scenario 3: Training Wrapper (Mocked)
*   **ID**: S04-03
*   **Goal**: Verify that `pace_train` is called with correct arguments.
*   **Priority**: Medium.
*   **Steps**:
    1.  Mock `subprocess.run`.
    2.  Run `Trainer.train(dataset_path)`.
    3.  Assert command contains `--dataset test_dataset.pckl.gzip`.
    4.  Assert command contains `--initial_potential` if provided.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Training Workflow

  Scenario: Saving Datasets
    Given a list of 10 atoms with energy and forces
    When I save them to "data.pckl.gzip"
    Then the file should exist
    And reading it back should return 10 structures

  Scenario: Selecting Active Set
    Given a pool of 100 candidate structures
    And a request to select 5 best structures
    When I run the active set selector
    Then I should get exactly 5 structures
    And they should maximize the information metric (mocked)
```
