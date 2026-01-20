# Cycle 04: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 4.1: Delta Learning Data Prep
-   **Priority**: High
-   **Description**: Verify that the training data is correctly prepared with ZBL subtraction.
-   **Steps**:
    1.  Create a structure where two atoms are very close (0.8 A). DFT Energy is high (+50 eV).
    2.  Calculate ZBL Energy (e.g., +45 eV).
    3.  Run `DatasetBuilder`.
    4.  Inspect the output `.xyz`.
-   **Success Criteria**:
    -   The energy listed in the file is approx +5 eV ($50 - 45$).
    -   This ensures the ML model only learns the "correction", not the steep wall.

### Scenario 4.2: Force Masking Export
-   **Priority**: Medium
-   **Description**: Verify that force masks in the DB are correctly written to the training file.
-   **Steps**:
    1.  Create a structure with 10 atoms.
    2.  Set `force_mask = [1, 1, 1, 0, 0, ...]` (first 3 atoms active).
    3.  Save to DB.
    4.  Run `DatasetBuilder`.
-   **Success Criteria**:
    -   The output `.xyz` has a `force_weights` column.
    -   The values correspond to the mask (1.0 for active, 0.0 for masked).
    -   Pacemaker log confirms "using force weights".

### Scenario 4.3: Training a Simple Potential
-   **Priority**: Critical
-   **Description**: Train a potential on a toy dataset and verify it works.
-   **Steps**:
    1.  Use the 5-structure Strain dataset from Cycle 2 (DFT calculated in Cycle 1).
    2.  Train a potential with `pacemaker`.
    3.  Load the potential in ASE (`ACE1` calculator).
    4.  Calculate energy of one of the training structures.
-   **Success Criteria**:
    -   The predicted energy matches the DFT energy (within ~10 meV/atom).
    -   This confirms the full loop: DB -> Prep -> Train -> Load.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Active Learning Training

  Scenario: Preparing data with ZBL subtraction
    Given a database with DFT results
    And a ZBL calculator
    When the dataset is exported
    Then the output energy should be E_DFT - E_ZBL
    And the output forces should be F_DFT - F_ZBL

  Scenario: Respecting Force Masks
    Given a structure with boundary atoms masked
    When the dataset is exported for Pacemaker
    Then the boundary atoms should have a force weight of 0.0
    And the core atoms should have a force weight of 1.0

  Scenario: Training Execution
    Given a valid training set config
    When the Pacemaker wrapper is invoked
    Then a .yace potential file should be generated
    And the training log should report RMSE convergence
```
