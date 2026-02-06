# Cycle 08 UAT: The Quality Gate

## 1. Test Scenarios

### Scenario 8: Final Validation
**Priority**: High
**Objective**: Verify that the system prevents bad potentials from being released.

**Steps**:
1.  **Preparation**:
    *   Create a "Bad" potential (e.g., random noise).
    *   Create a "Good" potential (e.g., LJ).
2.  **Execution**:
    *   Run the Validator on both.
3.  **Verification**:
    *   **Bad Potential**: Should FAIL the EOS check (non-smooth curve) or Elastic check (unstable).
    *   **Good Potential**: Should PASS.
    *   **Report**: Open `validation_report.html`. Check for plots.

## 2. Behavior Definitions

```gherkin
Feature: Physical Validation

  Scenario: Checking mechanical stability
    GIVEN a trained potential
    WHEN the Validator computes the elastic constants
    THEN it must verify the Born stability criteria
    AND if the criteria are violated, the validation result should be "FAIL"
```
