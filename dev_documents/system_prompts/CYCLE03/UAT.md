# Cycle 03 UAT: Oracle & DFT Interface

## 1. Test Scenarios

### SCENARIO 01: Single Point Calculation (Mocked QE)
**Priority**: High
**Description**: Verify that the system can successfully execute a DFT calculation using a mock binary.
**Pre-conditions**: `mock_pw.py` is available and executable. Config points to it.
**Steps**:
1.  Initialise `DFTManager` with `command="python mock_pw.py"`.
2.  Pass a simple structure (e.g., H2 molecule).
3.  Call `compute()`.
4.  Check the returned structure.
**Expected Result**: The structure has `results['energy']`, `results['forces']` populated with values from the mock output.

### SCENARIO 02: Self-Healing on Convergence Failure
**Priority**: Medium
**Description**: Verify that the system retries with adjusted parameters upon failure.
**Pre-conditions**: A modified `mock_pw.py` that fails once then succeeds.
**Steps**:
1.  Configure `DFTManager` to use the flaky mock.
2.  Call `compute()`.
3.  Inspect logs.
**Expected Result**:
-   Log shows "Calculation failed. Retrying with mixing_beta=0.3...".
-   The calculation eventually succeeds.
-   The final result is valid.

## 2. Behaviour Definitions

```gherkin
Feature: DFT Calculation

  Scenario: Successful calculation
    Given a valid structure
    And a configured DFTManager pointing to a valid QE binary (or mock)
    When I request a calculation
    Then I should receive the structure with Energy, Forces, and Stress
    And the calculation status should be "Converged"

  Scenario: Periodic Embedding
    Given a large supercell (1000 atoms)
    And a list of 5 active atoms
    When I apply periodic embedding
    Then I should get 5 small structures (clusters)
    And each cluster should be small enough for DFT (< 100 atoms)
```
