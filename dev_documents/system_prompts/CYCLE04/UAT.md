# Cycle 04 UAT: Trainer & Pacemaker Interface

## 1. Test Scenarios

### SCENARIO 01: Dataset Conversion
**Priority**: High
**Description**: Verify that the system can correctly convert ASE/Structure objects to Pacemaker's pickle format.
**Pre-conditions**: A list of `Structure` objects with energy and forces.
**Steps**:
1.  Call `DataManager.save_dataset(structures, "test.pckl.gzip")`.
2.  Load "test.pckl.gzip" using pandas.
3.  Check columns and values.
**Expected Result**: The file exists and contains the correct data.

### SCENARIO 02: Training Execution (Dry Run)
**Priority**: Medium
**Description**: Verify that the Trainer can launch the training process.
**Pre-conditions**: A valid dataset file. Configured `PacemakerTrainer`.
**Steps**:
1.  Call `train(dataset="test.pckl.gzip")`.
2.  (Mock the actual `pace_train` binary to just touch the output file).
3.  Check if the output potential path is returned.
**Expected Result**: The function returns a valid path to a `.yace` file.

## 2. Behaviour Definitions

```gherkin
Feature: Potential Training

  Scenario: Generating input files
    Given a Trainer configuration with cutoff=5.0 and order=3
    When I prepare the training run
    Then an "input.yaml" file should be created
    And it should contain "cutoff: 5.0"
    And it should contain "b_order: 3"

  Scenario: Active Set Selection
    Given a pool of 100 candidate structures
    When I request an active set of size 10
    Then I should receive 10 structures
    And these structures should be the ones with highest distinctness (mock check)
```
