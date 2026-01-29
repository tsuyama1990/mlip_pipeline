# Cycle 05 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 5.1: Phonon Stability Check
*   **ID**: UAT-05-01
*   **Priority**: High
*   **Description**: The system identifies unstable structures via phonon analysis.
*   **Steps**:
    1.  Create a structure known to be unstable with a given potential (or mock the result).
    2.  Run the Validation Phase.
    3.  Check the report. It should show "Phonon Stability: FAIL" and display imaginary modes in the band structure plot.

### Scenario 5.2: Report Generation
*   **ID**: UAT-05-02
*   **Priority**: Medium
*   **Description**: A human-readable report is generated after validation.
*   **Steps**:
    1.  Run the Validation Phase on a trained potential.
    2.  Open `validation_report.html` in a web browser.
    3.  Verify it contains:
        *   RMSE metrics.
        *   Elastic constant table.
        *   EOS plot.
        *   Phonon band structure.

## 2. Behavior Definitions

```gherkin
Feature: Physical Validation

  Scenario: Validating a stable potential
    GIVEN a trained potential
    AND a known stable crystal structure
    WHEN the Validation Phase executes
    THEN the Elasticity Validator should return positive bulk modulus
    AND the Phonon Validator should find no imaginary frequencies
    AND the phase status should be "PASS"

  Scenario: Detecting instability
    GIVEN a potential that produces negative frequencies
    WHEN the Validation Phase executes
    THEN the phase status should be "FAIL" (or CONDITIONAL)
    AND the report should highlight the instability
```
