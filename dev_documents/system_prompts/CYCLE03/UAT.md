# Cycle 03: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Dataset Generation
**Priority**: High
**Goal**: Verify that ASE atoms can be correctly converted to Pacemaker's format.
**Procedure**:
1.  Create a list of ASE atoms with random energies and forces.
2.  Use `DatasetManager` to save them to `test_data.pckl.gzip`.
3.  Load the file using `pandas` and check if columns "energy", "forces" exist.
**Success Criteria**:
*   File is created.
*   Data integrity is preserved (values match).

### Scenario 2: Training Configuration Check
**Priority**: Medium
**Goal**: Verify that the generated `input.yaml` for Pacemaker contains the correct physics settings.
**Procedure**:
1.  Initialize `TrainingPhase` with a config specifying `ZBL` baseline.
2.  Trigger the config generation method.
3.  Inspect the output YAML.
**Success Criteria**:
*   YAML contains `potential: delta: true`.
*   YAML contains `pair_style: zbl` with correct atomic numbers.

## 2. Behavior Definitions

```gherkin
Feature: Trainer Module

  Scenario: Serialize Data
    GIVEN a list of atomic structures with labels
    WHEN the DatasetManager processes them
    THEN a compressed pickle file should be created
    AND the file should be readable by Pacemaker

  Scenario: Run Training (Mock)
    GIVEN a training dataset and a config
    WHEN the Trainer executes the training loop
    THEN it should call the "pace_train" command
    AND it should produce a "potential.yace" file
    AND it should copy the potential to the versioned directory
```
