# Cycle 06 UAT: Validation & kMC Integration

## 1. Test Scenarios

### Scenario 01: Phonon Stability Validation (Mock)
**Priority**: High
**Description**: Verify that the `Validator` can run a phonon calculation and correctly identify stability.
**Steps**:
1.  Create a python script `test_phonon.py`.
2.  Mock `phonopy.Phonopy.get_band_structure_dict` to return imaginary frequencies (negative eigenvalues).
3.  Run `Validator.check_phonons(structure)`.
**Expected Result**:
-   Returns `stable=False`.
-   Logs warning "Imaginary frequencies detected".

### Scenario 02: Equation of State Check
**Priority**: Medium
**Description**: Verify that the `Validator` computes the bulk modulus using Birch-Murnaghan EOS.
**Steps**:
1.  Create a python script `test_eos.py`.
2.  Mock `get_potential_energy` to return energies for different volumes (parabolic).
3.  Run `Validator.check_eos(structure)`.
**Expected Result**:
-   Returns valid `bulk_modulus`.
-   Plot generated: `eos_curve.png`.

### Scenario 03: HTML Report Generation
**Priority**: Medium
**Description**: Verify that `Validator` aggregates results into a readable HTML report.
**Steps**:
1.  Create a `ValidationResult` object with dummy metrics and plot paths.
2.  Run `ReportGenerator.generate(result, output_path="report.html")`.
3.  Inspect `report.html`.
**Expected Result**:
-   File exists.
-   Contains metrics tables and `<img>` tags for plots.

### Scenario 04: EON kMC Execution (Mock)
**Priority**: High
**Description**: Verify that `EONWrapper` correctly sets up and runs an EON simulation.
**Steps**:
1.  Create a python script `test_eon.py`.
2.  Mock `subprocess.run` for `eonclient`.
3.  Run `EONWrapper.run_search(initial_state, potential_path)`.
4.  Check for `config.ini` creation.
**Expected Result**:
-   `config.ini` contains `potential = script`.
-   `pace_driver.py` is present in the directory.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Validation & Advanced Dynamics

  Scenario: Validate Phonon Stability
    Given a trained potential
    When I request phonon validation
    Then the system should calculate the band structure
    And report if any imaginary frequencies exist

  Scenario: Calculate Equation of State
    Given a trained potential
    When I request EOS validation
    Then the system should compute the bulk modulus
    And generate an energy-volume curve plot

  Scenario: Run kMC Simulation
    Given a stable potential and initial state
    When I request an EON kMC search
    Then the system should generate the EON configuration
    And execute the EON client with the potential driver
```
