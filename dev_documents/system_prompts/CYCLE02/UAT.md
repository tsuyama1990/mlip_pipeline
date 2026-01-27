# Cycle 02 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario ID: UAT-02-01 (Data Pipeline)
**Priority**: High
**Description**: Verify that structures can be generated, labelled (with mock DFT data), and saved to a format Pacemaker can read.
**Steps**:
1.  Run the structure generator for "Aluminium".
2.  Inspect the output file (e.g., using `ase gui` or a notebook).
3.  Check if the structures look physically reasonable (not overlapping too much, unless high pressure).
4.  Run the database saver.
5.  Verify the file size indicates content.

### Scenario ID: UAT-02-02 (Training a Potential)
**Priority**: Critical
**Description**: Prove that the system can actually produce a `.yace` potential file from a dataset.
**Prerequisites**: Pacemaker installed in the environment.
**Steps**:
1.  Provide a small valid dataset (e.g., 50 structures).
2.  Run the `train` command via `mlip-auto`.
3.  Watch the logs for "Training started" and "Epoch 1".
4.  Wait for completion.
5.  Verify `potential.yace` exists.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: One-Shot Training

  Scenario: Generate and Train
    GIVEN I have configured the system for "Cu"
    WHEN I execute the generation and training pipeline
    THEN the system should create a dataset of at least 10 structures
    AND the system should invoke the Pacemaker trainer
    AND a "potential.yace" file should be present in the output directory
    AND the training logs should show decreasing loss
```
