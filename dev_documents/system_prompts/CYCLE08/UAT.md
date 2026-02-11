# Cycle 08 User Acceptance Test (UAT)

## 1. Test Scenarios

### Scenario 1: Successful Validation (Mock)
**Priority**: P0 (Critical)
**Description**: Verify that the Validator accepts a "good" potential and generates a pass report.
**Steps**:
1.  Configure `config.yaml` with `validator.type: mock_pass`.
2.  Run `mlip-runner validate config.yaml`.
3.  Inspect the output.
    -   Expected: "Validation Result: PASSED"
    -   Expected: "Phonon Stable: True"
    -   Expected: "Elastic Stable: True"
4.  Open `validation_report.html` (if generated) and verify it contains plots.

### Scenario 2: Failed Validation (Mock)
**Priority**: P1 (High)
**Description**: Verify that the Validator rejects a "bad" potential (e.g., imaginary phonons).
**Steps**:
1.  Configure `config.yaml` with `validator.type: mock_fail`.
2.  Run the validation.
3.  Inspect the output.
    -   Expected: "Validation Result: FAILED"
    -   Expected: "Reason: Imaginary frequencies detected at Gamma point."

### Scenario 3: Real Validation (Physics Check)
**Priority**: P2 (Medium)
**Description**: Verify that the implemented physics checks (EOS, Elastic) return reasonable values for a simple system (e.g., LJ Argon).
**Steps**:
1.  Use a simple LJ potential (built-in to ASE).
2.  Run the Validator on an Argon FCC crystal.
3.  Verify Bulk Modulus is positive and within 10% of literature values.
4.  Verify C11 > C12 (Born stability).

## 2. Behavior Definitions (Gherkin)

### Feature: Physical Validation

**Scenario**: Check Phonon Stability
    **Given** a potential that predicts negative eigenvalues in the dynamical matrix
    **When** the Validator runs the phonon check
    **Then** it should report "Unstable"
    **And** the validation status should be "Failed"

**Scenario**: Check Elastic Constants
    **Given** a cubic crystal
    **When** the Validator calculates C11, C12, C44
    **Then** it should verify the Born stability criteria (C11 - C12 > 0, etc.)
    **And** report the calculated Bulk Modulus

**Scenario**: Generate Report
    **Given** a completed validation run
    **When** the Reporter is invoked
    **Then** an HTML file should be created
    **And** it should contain the EOS curve plot
