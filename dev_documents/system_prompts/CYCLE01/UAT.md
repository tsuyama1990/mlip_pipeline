# Cycle 01: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1.1: System Initialization & Config Validation
-   **Priority**: High
-   **Description**: Verify that the system correctly initializes from a `config.yaml` and rejects invalid configurations.
-   **Steps**:
    1.  Create a valid `config.yaml` with all required paths.
    2.  Run `mlip-auto check-config config.yaml`.
    3.  Create an invalid config (e.g., negative cutoff, missing paths).
    4.  Run check command again.
-   **Success Criteria**:
    -   Valid config returns "OK".
    -   Invalid config raises a clear Pydantic validation error explaining exactly which field failed.

### Scenario 1.2: Static DFT Calculation (Happy Path)
-   **Priority**: Critical
-   **Description**: Run a static calculation on a simple Silicon structure.
-   **Steps**:
    1.  Prepare an `input.xyz` with a 2-atom Si primitive cell.
    2.  Use the `QERunner` to execute an SCF calculation.
    3.  Inspect the database.
-   **Success Criteria**:
    -   Process completes with exit code 0.
    -   Database contains 1 row.
    -   Row has `energy` (approx -21.0 eV/atom range), `forces` (near zero for equilibrium), and `stress`.
    -   Metadata includes `calculation_type="scf"`.

### Scenario 1.3: Magnetism Auto-Detection
-   **Priority**: Medium
-   **Description**: Verify that Iron (Fe) automatically triggers spin-polarized calculation.
-   **Steps**:
    1.  Prepare an `input.xyz` with BCC Iron.
    2.  Run `QERunner`.
    3.  Inspect the generated `pw.x` input file (or logs).
-   **Success Criteria**:
    -   The input file contains `nspin = 2`.
    -   The input file contains `starting_magnetization(x) > 0`.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Automated DFT Execution

  Background:
    Given a working installation of Quantum Espresso (pw.x)
    And a valid pseudopotential directory

  Scenario: Running a standard SCF calculation
    Given an atomic structure "Al_fcc" loaded from file
    And a DFT configuration with "kpoints_density=0.15"
    When the QERunner executes the calculation
    Then the input file should contain "calculation = 'scf'"
    And the input file should contain "tprnfor = .true."
    And the process should exit successfully
    And the result should contain valid "energy", "forces", and "stress"

  Scenario: Saving results to database
    Given a completed DFT result object
    When the DatabaseManager adds the result
    Then the ASE database should show a count of 1
    And the stored atoms should have the "dft_scf" tag
```
