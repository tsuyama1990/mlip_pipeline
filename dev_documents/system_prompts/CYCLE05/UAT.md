# Cycle 05: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 01: Validation of a Good Potential
*   **ID**: UAT-CY05-01
*   **Priority**: High
*   **Description**: Verify that a physically sound potential passes all validation checks.
*   **Pre-conditions**: A high-quality potential (or a robust empirical potential like EAM masked as ACE).
*   **Steps**:
    1.  Run `mlip-auto validate --potential good.yace --structure Al.cif`.
*   **Expected Result**:
    *   Phonon dispersion shows no imaginary frequencies.
    *   Elastic constants satisfy Born stability conditions.
    *   Bulk modulus matches literature/reference values (within tolerance).
    *   Final result: "VALIDATION PASSED".

### Scenario 02: Rejection of Unstable Potential
*   **ID**: UAT-CY05-02
*   **Priority**: High
*   **Description**: Verify that the Validator catches physical instabilities.
*   **Pre-conditions**: A "bad" potential (e.g., undertrained, or trained only on high-T liquid data and tested on crystal).
*   **Steps**:
    1.  Run `mlip-auto validate --potential bad.yace --structure Al.cif`.
*   **Expected Result**:
    *   Phonon test detects imaginary modes (Soft modes).
    *   Or Elastic test shows violation of stability criteria (e.g., C11 - C12 < 0).
    *   Final result: "VALIDATION FAILED".
    *   Report highlights the specific failure mode.

### Scenario 03: Report Generation
*   **ID**: UAT-CY05-03
*   **Priority**: Medium
*   **Description**: Verify that the system generates a human-readable report.
*   **Steps**:
    1.  Run validation.
    2.  Open `validation_report.html` (or `.md`).
*   **Expected Result**:
    *   The report contains the Phonon Band Structure plot.
    *   The report contains the Energy vs Volume curve.
    *   The report lists calculated Cij values.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Physics Validation

  Scenario: Phonon Stability Check
    GIVEN a trained potential
    WHEN I calculate the phonon dispersion for the ground state structure
    AND there are imaginary frequencies (negative eigenvalues) at points other than Gamma
    THEN the validation status should be 'FAIL'
    AND the report should indicate 'Dynamic Instability detected'

  Scenario: Elastic Constants Check
    GIVEN a cubic crystal structure
    WHEN I calculate the elastic stiffness tensor Cij
    AND the condition (C11 - C12 > 0) is violated
    THEN the validation status should be 'FAIL'
    AND the report should indicate 'Mechanical Instability detected'
```
