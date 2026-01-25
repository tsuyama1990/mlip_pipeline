# Cycle 06 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 6.1: Validation Pass
*   **Priority:** High
*   **Description:** A physically sound potential passes all validation checks.
*   **Input:** A stable Lennard-Jones potential for Argon.
*   **Expected Output:**
    *   Phonon check: PASS (No imaginary freq).
    *   Elastic check: PASS (Born criteria met).
    *   EOS check: PASS (Positive Bulk Modulus).
    *   Overall Status: GREEN.

### Scenario 6.2: Detecting Instability (Phonon)
*   **Priority:** High
*   **Description:** A potential that is unstable (e.g., trained on insufficient data) should be flagged.
*   **Input:** A dummy potential that yields negative forces for small displacements.
*   **Expected Output:**
    *   Phonon check: FAIL (Imaginary frequencies detected).
    *   Overall Status: RED.

### Scenario 6.3: Reporting
*   **Priority:** Low
*   **Description:** The system generates a readable HTML report.
*   **Input:** Validation results.
*   **Expected Output:**
    *   `report.html` exists.
    *   Contains plots of EOS and Phonon bands.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Physical Validation

  Scenario: Stable Potential
    Given a robust potential
    When I run the validation suite
    Then the phonon dispersion should have no imaginary modes
    And the elastic constants should satisfy Born criteria
    And the Bulk Modulus should be positive

  Scenario: Unstable Potential
    Given a potential with imaginary phonon modes
    When I run the validation suite
    Then the validation result should be "FAIL"
    And the report should highlight the instability
```
