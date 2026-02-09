# Cycle 04 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Basic Training (Mocked)
**Priority**: High
**Goal**: Verify the training pipeline integration.
**Procedure**:
1.  Configure `trainer: max_epochs: 10`.
2.  Provide a small labeled dataset (mocked).
3.  Run the Trainer.
**Expected Result**:
*   The system generates an `input.yaml` configuration file for Pacemaker.
*   The system calls `pace_train`.
*   A `potential.yace` file is produced (mocked).

### Scenario 2: Active Set Selection
**Priority**: Medium
**Goal**: Verify data filtering.
**Procedure**:
1.  Provide a dataset with 100 identical structures.
2.  Configure `active_set: method: maxvol, count: 10`.
3.  Run the Trainer.
**Expected Result**:
*   The system calls `pace_activeset`.
*   The training log indicates that only a subset (approx 10) of structures were used for fitting.

### Scenario 3: Delta Learning Configuration
**Priority**: Medium
**Goal**: Verify that physics baseline is correctly configured.
**Procedure**:
1.  Configure `physics_baseline: type: lj`.
2.  Run the Trainer.
3.  Inspect the generated `input.yaml`.
**Expected Result**:
*   The YAML file contains the `delta` potential definition.
*   The YAML file defines the Lennard-Jones parameters.

## 2. Behavior Definitions

```gherkin
Feature: Potential Training

  Scenario: Training with a new dataset
    GIVEN a labeled dataset "data/train.pckl"
    AND a configuration specifying "delta-learning"
    WHEN "train" is executed
    THEN a Pacemaker input YAML should be generated
    AND "pace_train" should be called
    AND a "potential.yace" file should be created in "potentials/"

  Scenario: Updating an existing potential
    GIVEN an existing potential "generation_001.yace"
    AND new labeled data
    WHEN "train" is executed
    THEN the training should initialize weights from "generation_001.yace" (Fine-tuning)
    AND the training time should be shorter than training from scratch
```
