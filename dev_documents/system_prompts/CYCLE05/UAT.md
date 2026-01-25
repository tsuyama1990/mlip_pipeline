# Cycle 05 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Candidate Generation & Selection
*   **ID**: UAT-05-01
*   **Priority**: High
*   **Description**: Verify that the system generates a cloud of candidates around a halted structure and selects the best ones.
*   **Success Criteria**:
    *   Input: 1 Halted structure.
    *   Config: Generate 20 perturbations, Select 5.
    *   Output: 5 structures (likely including the original) ready for DFT.

### Scenario 2: Data Lineage Tracking
*   **ID**: UAT-05-02
*   **Priority**: Medium
*   **Description**: Verify that we can trace back a training structure to its origin.
*   **Success Criteria**:
    *   Query the database for a specific structure ID.
    *   Result shows `origin="perturbation"`, `parent_id=...`, and `status="trained"`.

### Scenario 3: Full Active Learning Loop (Mocked)
*   **ID**: UAT-05-03
*   **Priority**: Critical
*   **Description**: Verify the transition between all states in the workflow.
*   **Success Criteria**:
    *   Start in `EXPLORATION`. Mock a halt.
    *   System moves to `SELECTION`. Candidates generated.
    *   System moves to `CALCULATION`. Mock DFT success.
    *   System moves to `TRAINING`. Mock Training success.
    *   System moves back to `EXPLORATION` (Iteration counter incremented).

## 2. Behavior Definitions

```gherkin
Feature: Active Learning Strategy

  As a system architect
  I want the system to intelligently select training data
  So that we minimize expensive DFT calculations while maximizing potential quality

  Scenario: Handling a Simulation Halt
    GIVEN a simulation halted at step 5000 due to uncertainty
    WHEN the workflow processes this event
    THEN 20 candidate structures should be generated around the halted configuration
    AND the 5 most informative candidates should be sent to the Oracle
    AND these 5 structures should be logged in the database as "pending_dft"

  Scenario: Cycle Completion
    GIVEN the DFT calculations are finished
    WHEN the workflow enters the Refinement phase
    THEN the new data should be added to the training set
    AND the potential should be retrained
    AND the iteration counter should increase by 1
```
