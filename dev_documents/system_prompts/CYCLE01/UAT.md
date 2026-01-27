# Cycle 01 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario ID: UAT-01-01 (Configuration Validation)
**Priority**: High
**Description**: Verify that the system correctly loads valid configurations and rejects invalid ones with clear error messages.
**Steps**:
1.  Create a `config.yaml` with missing `pseudopotential_dir`.
2.  Run the system.
3.  Expect a validation error.
4.  Correct the path.
5.  Run again.
6.  Expect success.

### Scenario ID: UAT-01-02 (Oracle Dry Run)
**Priority**: Critical
**Description**: Perform a "Dry Run" of the Oracle module. This ensures that the system can interface with the Quantum Espresso binary (or a mock) and process results.
**Prerequisites**: A mock script mimicking `pw.x` or an actual installation of QE.
**Steps**:
1.  Define a simple Silicon unit cell in a Python script or structure file.
2.  Execute the `QERunner`.
3.  Check if an output directory is created.
4.  Check if `energy` and `forces` are extracted and printed.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: DFT Oracle

  Scenario: Run a static calculation on Silicon
    GIVEN a configuration file "config.yaml" pointing to valid pseudopotentials
    AND a structure file "Si.cif"
    WHEN I run the command "mlip-auto run-dft --config config.yaml --structure Si.cif"
    THEN the system should generate a Quantum Espresso input file "pw.in"
    AND the input file should contain "tprnfor = .true."
    AND the system should execute the DFT command
    AND the output should contain valid Energy and Force values
```
