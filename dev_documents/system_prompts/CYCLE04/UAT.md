# Cycle 04 UAT: The Trainer

## 1. Test Scenarios

### Scenario 01: Training from Scratch
**Priority**: High
**Description**: Train a potential from a small XYZ dataset.
**Objective**: Verify the training pipeline.

**Steps**:
1.  Prepare `train.xyz` with 50 frames of LJ-computed data (so we know the ground truth).
2.  Run the Trainer module: `python -m mlip_autopipec train config.yaml --data train.xyz`
3.  **Expected Result**:
    -   Logs show `pace_collect` execution.
    -   Logs show `pace_train` progress (Epoch 1..10).
    -   A file `potentials/generation_001.yace` is generated.
    -   The final RMSE printed in logs is reasonable (e.g., < 10 meV/atom for LJ data).

### Scenario 02: Active Set Selection
**Priority**: Medium
**Description**: Filter a redundant dataset.
**Objective**: Verify D-Optimality.

**Steps**:
1.  Prepare `redundant.xyz` where 50 frames are identical copies of the first frame.
2.  Call `trainer.select_active_set("redundant.xyz")`.
3.  **Expected Result**:
    -   The output dataset contains significantly fewer structures (ideally close to 1, or small number depending on B-basis size).
    -   Logs indicate "Selected X structures out of 50".

### Scenario 03: Validation Failure
**Priority**: Low
**Description**: Train on garbage data.
**Objective**: Verify error reporting.

**Steps**:
1.  Prepare `garbage.xyz` with random energies unrelated to positions.
2.  Run training.
3.  **Expected Result**:
    -   Training might finish, but Validation step should flag "High RMSE".
    -   Orchestrator logs a warning: `[WARN] Potential quality below threshold. RMSE: 500 meV/atom.`

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Potential Training

  Scenario: Basic Training
    Given a dataset of 100 labeled structures
    And a training configuration with "max_generations=100"
    When the Trainer executes
    Then a ".yace" potential file should be created
    And the training log should show decreasing loss

  Scenario: Active Set Optimization
    Given a candidate pool of 1000 structures
    And many structures are geometrically similar
    When Active Set selection is applied
    Then the reduced dataset size should be significantly smaller than 1000
    And the information content (D-optimality) should be maximized

  Scenario: Hybrid Potential Configuration
    Given a configuration specifying "ZBL" as reference
    When the training input is generated
    Then the Pacemaker configuration should include the "ZBL" embedding section
```
