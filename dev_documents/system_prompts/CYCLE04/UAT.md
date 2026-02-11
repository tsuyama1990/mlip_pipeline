# Cycle 04 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 4.1: Dataset Creation
*   **Priority**: High
*   **Goal**: Verify data can be prepared for Pacemaker.
*   **Action**:
    1.  Provide 10 `ase.Atoms` objects with energies/forces.
    2.  Invoke `dataset_utils.create_dataset("test.pckl.gzip")`.
*   **Expectation**:
    *   File `test.pckl.gzip` is created.
    *   It is a valid pandas DataFrame readable by `pandas.read_pickle`.
    *   Columns include `energy`, `forces`, `ASE_atoms`.

### Scenario 4.2: Pacemaker Config Generation
*   **Priority**: Medium
*   **Goal**: Verify `input.yaml` correctness.
*   **Action**:
    1.  Configure `training.max_epochs: 50`.
    2.  Invoke `PacemakerWrapper.generate_config()`.
*   **Expectation**:
    *   Generated YAML string contains `max_num_epochs: 50`.
    *   It contains the correct path to the dataset.

### Scenario 4.3: Training Execution (Mock)
*   **Priority**: Medium
*   **Goal**: Verify workflow integration.
*   **Action**:
    1.  Configure `training.type: MOCK`.
    2.  Run `Orchestrator` cycle 3 (Training phase).
*   **Expectation**:
    *   Log file shows "Mock Training started".
    *   File `potentials/generation_001.yace` is created (dummy file).
    *   Orchestrator state updates `current_potential_path`.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Potential Training

  Scenario: Trainer prepares data
    Given a list of new labeled structures
    When the Trainer updates the dataset
    Then the new structures should be appended to "train.pckl.gzip"
    And the total count of structures should increase

  Scenario: Trainer runs optimization
    Given a training dataset and configuration
    When the Trainer executes
    Then a new potential file "potential.yace" should be generated
    And the training log should be saved
```
