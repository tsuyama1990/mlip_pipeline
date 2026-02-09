# Cycle 08 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

These tests verify the Validator's ability to assess potential quality.

### Scenario 8.1: Phonon Stability Check
**Objective**: Ensure the potential predicts a dynamically stable lattice.
**Priority**: High (P1)

*   **Setup**: A known stable crystal (e.g., MgO ground state) and a reasonable potential.
*   **Action**: Run `validators.phonons.calculate(structure, potential)`.
*   **Expected Outcome**:
    *   Phonon band structure is calculated.
    *   No significant imaginary frequencies ($\omega < -0.1$ THz).
    *   A plot `phonon_band.png` is generated.
    *   `is_stable` returns True.

### Scenario 8.2: Elastic Constants & Born Stability
**Objective**: Verify mechanical stability criteria.
**Priority**: High (P1)

*   **Setup**: Cubic crystal (MgO).
*   **Action**: Run `validators.elastic.calculate(structure, potential)`.
*   **Expected Outcome**:
    *   Elastic tensor $C_{ij}$ is computed.
    *   $C_{11} - C_{12} > 0$.
    *   $C_{44} > 0$.
    *   $C_{11} + 2C_{12} > 0$.
    *   `is_stable` returns True.

### Scenario 8.3: EOS Fit (Thermodynamics)
**Objective**: Verify equilibrium volume and bulk modulus.
**Priority**: Medium (P2)

*   **Setup**: Start with a slightly expanded structure ($V = 1.05 V_0$).
*   **Action**: Run `validators.eos.calculate(structure, potential)`.
*   **Expected Outcome**:
    *   Energy-Volume curve is parabolic-like.
    *   Minimum energy volume $V_{min}$ is found.
    *   Bulk Modulus $B_0$ matches literature/DFT within tolerance.
    *   A plot `eos_fit.png` is generated.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Potential Validation

  Scenario: Checking dynamical stability
    Given a trained potential for "MgO"
    When I calculate the phonon dispersion relation
    Then there should be no imaginary frequencies in the Brillouin Zone
    And the system should be marked as dynamically stable

  Scenario: Generating a validation report
    Given a set of validation results (Phonons, Elastic, EOS)
    When I generate the final report
    Then an HTML file "validation_report.html" should be created
    And the file should contain plots for phonon bands and EOS curves
    And the file should list the calculated elastic constants
```
