# Cycle 05 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 05-01: Detecting Unstable Potentials
- **Priority**: High
- **Description**: Verify that the system correctly identifies and rejects a physically unstable potential.
- **Steps**:
    1. Create/Train a potential on a very small dataset (likely unstable).
    2. Run the validation tool.
    3. **Expected Result**: The report shows "Phonon Stability: FAILED" (Imaginary frequencies detected). The overall status is "REJECTED".

### Scenario 05-02: Elastic Constant Accuracy
- **Priority**: Medium
- **Description**: Verify that the calculated elastic constants match known values for a reference potential.
- **Steps**:
    1. Use a standard EAM potential (e.g., Al_mishin) wrapped as the "candidate".
    2. Run the elastic validation.
    3. **Expected Result**: Calculated $C_{11}, C_{12}, C_{44}$ match literature values within 5%.

### Scenario 05-03: Automated Report Generation
- **Priority**: Low
- **Description**: Verify that the user receives a readable report.
- **Steps**:
    1. Run validation on a valid potential.
    2. Open the generated `report.html` (or view `report.json`).
    3. **Expected Result**: The file contains plots for Phonon dispersion and EOS curves, and a summary table of errors.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Physics Validation

  Scenario: Validating a stable potential
    GIVEN a fully trained potential
    WHEN I run the validation suite
    THEN it should compute phonon dispersion curves
    AND it should verify that no imaginary frequencies exist
    AND it should confirm the bulk modulus is positive

  Scenario: Gating deployment
    GIVEN a potential that fails Born stability criteria
    WHEN the orchestrator receives the validation report
    THEN it should NOT mark the potential as "Production Ready"
    AND it should trigger an alert or a new sampling phase
```
