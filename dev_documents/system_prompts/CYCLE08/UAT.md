# Cycle 08 UAT: Validation & Reporting

## 1. Test Scenarios

### Scenario 1: Phonon Stability Check
*   **ID**: S08-01
*   **Goal**: Verify that unstable crystals are flagged.
*   **Priority**: Critical.
*   **Steps**:
    1.  Create a dummy potential for a known stable crystal (e.g., Al fcc).
    2.  Mock `phonopy` (or run if available).
    3.  Run `PhononValidator.validate(structure, potential)`.
    4.  Assert `result.stable` is True.
    5.  Repeat with distorted structure (mocked instability).
    6.  Assert `result.stable` is False.

### Scenario 2: Elastic Constants
*   **ID**: S08-02
*   **Goal**: Verify that C11, C12, C44 are calculated.
*   **Priority**: High.
*   **Steps**:
    1.  Use a Lennard-Jones potential (analytical values known).
    2.  Run `ElasticValidator.validate()`.
    3.  Assert `C11` is within 10% of analytical value.
    4.  Assert `Born criteria` satisfied.

### Scenario 3: Report Generation
*   **ID**: S08-03
*   **Goal**: Verify that the HTML report is generated with content.
*   **Priority**: Medium.
*   **Steps**:
    1.  Mock `TrainingMetrics` (RMSE list).
    2.  Mock `ValidationResult`.
    3.  Run `ReportGenerator.generate(metrics, validation)`.
    4.  Check `report.html` existence.
    5.  Assert content contains "RMSE Energy" and "Phonon Stability".

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Validation Suite

  Scenario: Checking Phonon Stability
    Given a trained potential
    And a target crystal structure
    When I validate phonon stability
    Then the result should be "Stable" or "Unstable"
    And imaginary frequencies should be reported if any

  Scenario: Generating HTML Report
    Given training metrics and validation results
    When I request a report
    Then a file "report.html" should be created
    And it should contain plots of RMSE vs Epoch
```
