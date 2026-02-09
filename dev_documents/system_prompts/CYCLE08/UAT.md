# Cycle 08 UAT: Validator & Reporting

## 1. Test Scenarios

### Scenario 8.1: Basic EOS Validation
*   **Goal:** Verify that the system can calculate the Equation of State (EOS).
*   **Steps:**
    1.  Provide a valid `Structure` (equilibrium bulk).
    2.  Run `EOSCalculator.fit()`.
    3.  Inspect the result.
*   **Expected Behavior:**
    *   $V_0$ (equilibrium volume) matches the input structure.
    *   $B_0$ (Bulk Modulus) is positive and reasonable (e.g., 50-200 GPa).
    *   $B'_0$ is positive (typically ~4).

### Scenario 8.2: Phonon Stability Check
*   **Goal:** Verify that the system detects unstable phonon modes.
*   **Steps:**
    1.  Provide a dynamically unstable structure (e.g., high-temp phase at 0K).
    2.  Run `PhononCalculator.check()`.
    3.  Check the boolean result and imaginary frequencies.
*   **Expected Behavior:**
    *   Result: `False` (Unstable).
    *   Max Imaginary Frequency: Significant negative value (e.g., < -1 THz).

### Scenario 8.3: Full Validation Report
*   **Goal:** Verify that an HTML report is generated with all metrics.
*   **Steps:**
    1.  Run the full `Validator.validate()` pipeline on a test potential.
    2.  Open `validation_report.html` in a browser (or check file content).
*   **Expected Behavior:**
    *   The file exists.
    *   Contains sections: "Summary", "EOS", "Elastic Constants", "Phonons".
    *   Contains embedded plots (PNG data).
    *   Shows a clear PASS/FAIL verdict.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Potential Validation

  Scenario: Calculate Equation of State
    Given a bulk crystal structure
    When I compute the energy-volume curve
    Then the Birch-Murnaghan fit should converge
    And the bulk modulus should be physically reasonable

  Scenario: Verify Dynamical Stability
    Given a potential and a structure
    When I calculate phonon dispersion
    Then the system should report if imaginary frequencies exist
    And the band structure plot should be generated

  Scenario: Generate Validation Report
    Given a set of validation results (EOS, Elastic, Phonon)
    When I generate the report
    Then an HTML file should be created
    And it should summarize the potential's quality with a Pass/Fail grade
```
