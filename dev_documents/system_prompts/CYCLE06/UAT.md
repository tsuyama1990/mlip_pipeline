# Cycle 06 UAT: Validation & Production Readiness

## 1. Test Scenarios

### Scenario 06: "The Final Exam"
**Priority**: High
**Description**: Verify that the Validator correctly accepts a good potential and rejects a bad one (e.g., one that is dynamically unstable). Generate a report for the user.

**Pre-conditions**:
-   A trained `potential.yace`.
-   `phonopy` (optional) and `matplotlib` installed.

**Steps**:
1.  User creates a `config.yaml` with `validation.max_rmse_force: 0.01`.
2.  User runs `pyacemaker validate --potential potential.yace` (New CLI command).
3.  User inspects `validation_report.html`.

**Expected Outcome**:
-   Console output summarizes Pass/Fail status.
-   HTML report contains Parity Plots and Phonon Dispersion curves (if phonopy present).
-   If potential is unstable (imaginary modes), the validation fails.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Potential Validation

  Scenario: Validate a Good Potential
    Given a potential with low RMSE
    And a stable crystal structure
    When I run the validation suite
    Then the result should be "PASSED"
    And an HTML report should be generated

  Scenario: Reject Unstable Potential
    Given a potential that predicts imaginary phonons
    When I run the validation suite
    Then the result should be "FAILED"
    And the report should highlight "Dynamic Instability"
```
