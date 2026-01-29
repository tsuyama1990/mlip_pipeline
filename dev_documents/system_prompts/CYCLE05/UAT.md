# User Acceptance Testing (UAT): Cycle 05

## 1. Test Scenarios

Cycle 05 ensures the potential is physically meaningful.

### Scenario 5.1: Phonon Stability Check
-   **ID**: UAT-C05-01
-   **Priority**: High
-   **Description**: Verify that a stable crystal structure (like FCC Al) has no imaginary phonon modes.
-   **Success Criteria**:
    -   Run validation on a known good potential.
    -   Phonon band structure plot is generated.
    -   No frequencies drop below zero (within small numerical tolerance).
    -   Status is "PASS".

### Scenario 5.2: Catching a Bad Potential
-   **ID**: UAT-C05-02
-   **Priority**: Medium
-   **Description**: Force the validation to fail by testing an unstable structure or a bad potential.
-   **Success Criteria**:
    -   The system identifies imaginary frequencies.
    -   The `overall_status` is "FAIL".
    -   The HTML report highlights the failure in red.

### Scenario 5.3: HTML Report Generation
-   **ID**: UAT-C05-03
-   **Priority**: Medium
-   **Description**: The user wants to see the results.
-   **Success Criteria**:
    -   A file `validation_report.html` is created.
    -   It opens in a standard web browser.
    -   It contains the Phonon Dispersion Plot (image).
    -   It contains a table of Elastic Constants.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Physical Validation of Potentials

  Background:
    Given a trained potential "potential.yace"
    And a configuration for "Silicon"

  Scenario: Validate a stable potential
    When I run the "mlip-auto validate" command
    Then the system should calculate Phonons
    And the system should calculate Elastic Constants
    And the validation result should be "PASS"
    And an HTML report should be generated

  Scenario: Detect unstable phonons
    Given a potential that predicts negative hessian eigenvalues
    When I run the validation
    Then the Phonon check should fail
    And the report should warn about "Imaginary Frequencies"

  Scenario: Check Bulk Modulus
    Given the experimental Bulk Modulus is 98 GPa
    When I run the EOS validation
    Then the calculated Bulk Modulus should be within 15% of 98 GPa
```
