# Cycle 08 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Physical Validation of MgO
**Priority**: High
**Goal**: Verify the potential produces correct physics.
**Procedure**:
1.  Train a potential on MgO (Cycle 01-04).
2.  Configure Validator: `phonon: true, elastic: true`.
3.  Run Validation.
**Expected Result**:
*   `phonon_stable` is True.
*   `bulk_modulus` is around 160 GPa (Literature value).
*   Report shows the Phonon Dispersion curve.

### Scenario 2: Report Generation
**Priority**: Medium
**Goal**: Verify usability of outputs.
**Procedure**:
1.  Complete a full run.
2.  Open `report.html`.
**Expected Result**:
*   The file opens in a browser.
*   Plots (RMSE, Data Count) are visible.
*   A table summarizes the final potential quality.

### Scenario 3: End-to-End Stress Test
**Priority**: Critical
**Goal**: The "Final Exam".
**Procedure**:
1.  Run the full pipeline on a standard Linux workstation.
2.  Use `IS_CI_MODE=False` (Real execution).
3.  Wait for completion.
**Expected Result**:
*   No crashes.
*   Final potential passes all checks.
*   Ready for scientific use.

## 2. Behavior Definitions

```gherkin
Feature: Validation and Reporting

  Scenario: Validating a stable crystal
    GIVEN a trained potential for Silicon
    WHEN the "Phonon Calculator" runs
    THEN it should not detect imaginary frequencies
    AND the "Elastic Calculator" should return positive elastic constants

  Scenario: Generating the final report
    GIVEN a completed active learning campaign
    WHEN "ReportGenerator" is executed
    THEN it should compile metrics from all cycles
    AND produce an HTML summary with interactive plots
```
