# Cycle 08: Validator & Quality Assurance - UAT

## 1. Test Scenarios

### Scenario 1: Basic Validation (EOS & Elasticity)
*   **ID**: UAT-08-001
*   **Objective**: Ensure the `StandardValidator` can run physical tests.
*   **Pre-conditions**: A valid potential file.
*   **Steps**:
    1.  Configure `ValidatorConfig` with `eos_strain=0.1`.
    2.  Run `validator.validate(potential)`.
    3.  Check the result.
*   **Expected Result**:
    *   Returns `ValidationResult`.
    *   `result.eos_params` (V0, B0) are physically reasonable ($B_0 > 0$).
    *   `result.elastic_constants` ($C_{11}, C_{12}$) satisfy stability criteria ($C_{11}-C_{12} > 0$).

### Scenario 2: Phonon Stability Check
*   **ID**: UAT-08-002
*   **Objective**: Detect imaginary frequencies in unstable structures.
*   **Pre-conditions**: A potential that predicts an unstable phase (or a mock that returns imaginary modes).
*   **Steps**:
    1.  Configure `ValidatorConfig` with `phonon_displacement=0.01`.
    2.  Run `validator.validate(potential)`.
*   **Expected Result**:
    *   The validation fails (`passed=False`).
    *   The reason includes "Imaginary phonon modes detected".
    *   The report includes a plot showing negative frequencies.

### Scenario 3: Validation Report Generation
*   **ID**: UAT-08-003
*   **Objective**: Verify the HTML report is generated correctly.
*   **Pre-conditions**: Validation run completed.
*   **Steps**:
    1.  Check the output directory for `validation_report.html`.
    2.  Open the file (or check its content).
*   **Expected Result**:
    *   The file exists and contains valid HTML.
    *   It contains sections for "Summary", "EOS", "Phonons", "Elasticity".
    *   It contains embedded base64 images of plots.

## 2. Behavior Definitions

```gherkin
Feature: Potential Quality Assurance

  Scenario: Validating a Good Potential
    Given I have a well-trained potential for a stable crystal
    When I run the validation suite
    Then the Equation of State should yield a positive Bulk Modulus
    And the Elastic Constants should satisfy the Born stability criteria
    And the Phonon Dispersion should have no imaginary modes
    And the validation result should be "PASS"

  Scenario: Rejecting an Unstable Potential
    Given I have a potential that predicts unphysical forces
    When I run the validation suite
    Then the Phonon Dispersion should show imaginary frequencies
    And the validation result should be "FAIL"
    And the report should highlight the instability
```
