# Cycle 01: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 01: Configuration Validation
*   **ID**: UAT-CY01-01
*   **Priority**: High
*   **Description**: Verify that the system correctly parses a valid configuration file and rejects invalid ones with helpful error messages. This ensures the "Zero-Config" promise starts with a robust entry point.
*   **Pre-conditions**: The `mlip-auto` CLI is installed.
*   **Steps**:
    1.  Create a file `invalid_config.yaml` with missing required fields (e.g., missing `pseudopotential_dir`).
    2.  Run `mlip-auto validate --config invalid_config.yaml`.
    3.  Create a file `valid_config.yaml` with all fields correct.
    4.  Run `mlip-auto validate --config valid_config.yaml`.
*   **Expected Result**:
    *   Step 2: Output should clearly state "Validation Error: Field 'pseudopotential_dir' required".
    *   Step 4: Output should state "Configuration Valid".

### Scenario 02: Single Point DFT Calculation (Mocked/Real)
*   **ID**: UAT-CY01-02
*   **Priority**: Critical
*   **Description**: Verify that the Oracle module can accept an atomic structure, generate the correct input files for Quantum Espresso, execute the command (or mock), and parse the resulting forces and energy.
*   **Pre-conditions**: A valid `structure.xyz` file exists. If running real DFT, `pw.x` must be in the PATH.
*   **Steps**:
    1.  Prepare `config.yaml` specifying the DFT parameters.
    2.  Run the test command: `mlip-auto test-oracle --config config.yaml --structure structure.xyz`.
    3.  Check the output logs.
*   **Expected Result**:
    *   The system creates a working directory (e.g., `tmp_dft/`).
    *   A `pw.in` file is generated containing the correct coordinates from `structure.xyz`.
    *   The system reports "Calculation Converged".
    *   The logs display the calculated Energy (eV) and Forces.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: DFT Oracle Execution

  Scenario: Run a successful static calculation
    GIVEN a configuration file with valid QE settings
    AND an atomic structure of a Silicon crystal
    WHEN I request a single-point calculation via the Oracle
    THEN the system should generate a 'pw.in' file
    AND the 'pw.in' file should contain 'tprnfor = .true.'
    AND the system should execute the QE command
    AND the result should contain Energy and Forces
    AND the result status should be 'converged'

  Scenario: Handle SCF convergence failure
    GIVEN a configuration file
    AND a structure that causes SCF issues (simulated)
    WHEN I request a single-point calculation
    AND the QE process returns a non-zero exit code OR the output file says "convergence NOT achieved"
    THEN the Oracle should mark the result as 'unconverged'
    AND the system should NOT crash
    AND the error should be logged for review
```
