# Cycle 03 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario 3.1: Dataset Accumulation
**Priority**: High
**Description**: Verify that the system correctly accumulates training data over multiple cycles.
**Steps**:
1.  Initialize the Trainer with an empty dataset.
2.  Add a batch of 10 structures (Cycle 1).
3.  Add another batch of 10 structures (Cycle 2).
4.  **Check**: The underlying `data.pckl.gzip` (or extxyz) should contain 20 structures.
5.  **Check**: Duplicate structures (same unique ID/hash) should be rejected or handled.

### Scenario 3.2: Delta Learning Configuration
**Priority**: Critical
**Description**: Verify that the system correctly configures ZBL/LJ baselines to prevent non-physical potentials.
**Steps**:
1.  Set `trainer.physics_baseline = "zbl"`.
2.  Run `trainer.prepare_input()`.
3.  Inspect the generated `input.yaml`.
4.  **Expectation**: The YAML must contain a section enabling ZBL repulsions with parameters appropriate for the elements (e.g., Z numbers).

### Scenario 3.3: Active Set Selection
**Priority**: Medium
**Description**: Verify that `pace_activeset` is called to optimize the basis.
**Steps**:
1.  Provide a large dataset (100 structures).
2.  Set `active_set_size = 50`.
3.  Run `trainer.select_active_set()`.
4.  **Expectation**: The trainer should invoke the `pace_activeset` command and reduce the effective number of basis configurations used for fitting.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Potential Training

  Scenario: Configuring ZBL Baseline
    Given the user requires "zbl" physics baseline
    When the trainer generates the input file
    Then the configuration should include "pair_style: zbl"
    And the ZBL parameters for the specific elements should be present

  Scenario: Successful Training Run
    Given a valid dataset and configuration
    When I call the train method
    Then the "pace_train" command should be executed
    And a "potential.yace" file should be produced
    And the method should return the path to this file
```
