# Cycle 05 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario 5.1: Pass/Fail Logic
**Priority**: Critical
**Description**: Verify that the Validator correctly rejects potentials that do not meet accuracy requirements.
**Steps**:
1.  Configure `energy_rmse_threshold = 1.0` (meV/atom).
2.  Provide a potential that yields 5.0 meV/atom error on the test set.
3.  Run `validator.validate()`.
4.  **Expectation**: `result.passed` is `False`, and `result.reason` cites "Energy RMSE too high".
5.  Relax the threshold to 10.0.
6.  Run `validator.validate()`.
7.  **Expectation**: `result.passed` is `True`.

### Scenario 5.2: Stability Crash Test
**Priority**: High
**Description**: Verify that the Validator detects potentials that cause simulations to explode.
**Steps**:
1.  Provide a "bad" potential (mocked or deliberately broken).
2.  Run the stability test (short MD).
3.  **Expectation**: The MD simulation should fail (or produce NaN forces).
4.  **Expectation**: The Validator should catch this and return `passed=False` with reason "Stability test failed".

### Scenario 5.3: Reporting
**Priority**: Medium
**Description**: Verify that a readable report is generated.
**Steps**:
1.  Run validation successfully.
2.  Check the output directory.
3.  **Check**: `validation_results.json` exists and contains the calculated metrics.
4.  **Check**: (Optional) `parity_plot.png` is generated showing correlation between DFT and MLIP forces.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Model Validation

  Scenario: Rejecting Inaccurate Models
    Given a maximum force RMSE of 0.05 eV/A
    When the model has an RMSE of 0.10 eV/A
    Then the validation result should be "Failed"

  Scenario: Detecting Instability
    Given a model that predicts infinite forces at short distances
    When I run the stability test
    Then the validator should report a crash
    And the model should be marked as "Unsafe"
```
