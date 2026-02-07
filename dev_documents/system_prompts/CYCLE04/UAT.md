# Cycle 04 User Acceptance Testing (UAT)

## 1. Test Scenarios

### SCENARIO 01: Training a Potential
**Priority**: High
**Goal**: Verify the system can train a valid potential file from a dataset.

**Steps**:
1.  Use the Dataset generated in Cycle 02.
2.  Configure `TrainerConfig`:
    ```yaml
    trainer:
      type: pacemaker
      max_num_epochs: 10
      batch_size: 2
    ```
3.  Run CLI: `mlip-pipeline train --config config.yaml`.
4.  Check `training_output/potential.yace`.
5.  Check logs for "Training completed in X seconds".

### SCENARIO 02: Active Set Optimization
**Priority**: Medium
**Goal**: Verify that the active set selection reduces the dataset size.

**Steps**:
1.  Provide 100 random structures.
2.  Configure `TrainerConfig` with `active_set_selection: True` and target size 10.
3.  Run the pipeline.
4.  Check that the training only used 10 structures (logs or output files).

## 2. Behavior Definitions

### Feature: Trainer Workflow
**Scenario**: Training with Active Set
  **Given** a dataset of 100 structures
  **And** a request to train using only the 10 most informative structures
  **When** the Trainer runs
  **Then** it should select the 10 structures maximizing the information matrix determinant (MaxVol)
  **And** train the potential using only these 10 structures
  **And** produce a valid `.yace` file
