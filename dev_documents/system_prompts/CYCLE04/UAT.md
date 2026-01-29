# Cycle 04 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 4.1: Dataset Creation and Active Set Selection
*   **ID**: UAT-04-01
*   **Priority**: Medium
*   **Description**: New DFT data is correctly added to the dataset and filtered.
*   **Steps**:
    1.  Mock the Oracle phase to return 10 `DFTResult`s.
    2.  Run the Training Phase.
    3.  Verify that a dataset file (e.g., `data/accumulated.pckl.gzip`) is created/updated.
    4.  Verify that the log mentions "Active Set Selection" (if enabled).

### Scenario 4.2: Potential Training Execution
*   **ID**: UAT-04-02
*   **Priority**: High
*   **Description**: The system successfully generates a `.yace` potential file.
*   **Steps**:
    1.  Ensure a valid dataset exists.
    2.  Run the Training Phase.
    3.  Verify that `pace_train` command was executed (check logs).
    4.  Check for the existence of `output_potential.yace` in the training directory.

## 2. Behavior Definitions

```gherkin
Feature: Pacemaker Training

  Scenario: Configuring Delta Learning
    GIVEN a config with "reference_potential: ZBL"
    WHEN the Trainer prepares the input
    THEN the generated "input.yaml" should contain the ZBL parameters
    AND the ACE potential should be set to learn the difference

  Scenario: Training Process
    GIVEN an updated dataset
    WHEN the Training Phase runs
    THEN it should invoke the "pace_train" command
    AND produce a new potential file
    AND update the workflow state with the new potential path
```
