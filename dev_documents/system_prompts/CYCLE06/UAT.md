# Cycle 06 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario ID: UAT-C06-001 - Training a Potential from Database

**Priority:** High
**Description:**
This scenario validates the "Learning" capability. The user takes the data generated in Cycle 04/05 and trains the first version of the potential. This ensures the data format is compatible with Pacemaker and the training loop executes correctly.

**User Story:**
As a Researcher, I want to click a "Train" button (or run a command) that automatically gathers all my DFT data, splits it into training and validation sets, runs the fitting code, and reports the final accuracy (RMSE), so I can see if my model is improving.

**Step-by-Step Walkthrough:**
1.  **Preparation**: The database contains 50 completed DFT calculations (Al).
2.  **Configuration**: User sets `training.max_iter: 100` (short run for test).
3.  **Execution**: User runs `mlip-auto train`.
    -   *Expectation*: CLI shows "Exporting data...", "Starting Pacemaker...", "Training...".
4.  **Completion**: The command finishes.
    -   *Output*: "Training Completed. Final RMSE (Energy): 5.2 meV/atom. Final RMSE (Force): 0.03 eV/A."
    -   *Artifacts*: A file `current_potential.yace` exists.
5.  **Verification**:
    -   User checks `training_data/train.xyz` exists.
    -   User checks `log.txt` for convergence plots.

**Success Criteria:**
-   The process runs end-to-end without crashing.
-   The produced `.yace` file is valid (not empty).
-   The reported RMSE is parsed correctly from the logs.

### Scenario ID: UAT-C06-002 - Training with Weighted Loss (Force-Centric)

**Priority:** Medium
**Description:**
For MD simulations, forces are more important than energy. We verify that the user can adjust weights.

**User Story:**
As an Expert User, I want to set the Force Weight to 100.0 and Energy Weight to 1.0, because I care about dynamics. I expect the resulting potential to have very low force error, even if energy error is slightly higher.

**Step-by-Step Walkthrough:**
1.  **Configuration**: `training.force_weight: 100.0`.
2.  **Execution**: `mlip-auto train`.
3.  **Verification**:
    -   Check the generated `input.yaml` for Pacemaker.
    -   *Expectation*: It should contain `weights: { energy: 1.0, forces: 100.0 }`.
    -   *Outcome*: The training log should reflect these weights in the loss function printout.

**Success Criteria:**
-   Configuration propagates correctly to the external tool's input file.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: MLIP Training
  As a Learning Engine
  I want to fit an ACE potential to DFT data
  So that I can use it for fast simulations

  Background:
    Given the database has 100 completed structures

  Scenario: Export Data for Training
    When I trigger the dataset builder
    Then a folder "training_data" should be created
    And "train.xyz" should contain approx 90 structures
    And "test.xyz" should contain approx 10 structures
    And the files should be in "extxyz" format

  Scenario: Execute Pacemaker Training
    Given the training data is ready
    And the configuration specifies "cutoff = 5.0"
    When the Pacemaker wrapper is called
    Then "input.yaml" should be generated with "cutoff: 5.0"
    And the "pacemaker" binary should be executed
    And the output should be logged to a file
    And "output.yace" should be produced upon success
```
