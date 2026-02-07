# Cycle 03 User Acceptance Testing (UAT)

## 1. Test Scenarios

### SCENARIO 01: DFT Calculation Success
**Priority**: High
**Goal**: Verify the system can run a Quantum Espresso calculation and return correct data.

**Steps**:
1.  Create a simple structure file `h2.xyz` (Hydrogen molecule).
2.  Configure `OracleConfig` to use `qe`:
    ```yaml
    oracle:
      type: qe
      command: pw.x
      pseudo_dir: /path/to/pseudos
      kspacing: 0.04
    ```
3.  Run CLI: `mlip-pipeline compute --config config.yaml --input h2.xyz --output h2_dft.xyz`.
4.  Check output `h2_dft.xyz`. It should contain `energy` (float) and `forces` (list of floats).

### SCENARIO 02: Self-Healing Logic
**Priority**: Medium
**Goal**: Verify the system can recover from a failed SCF cycle by adjusting parameters.

**Steps**:
1.  Mock the `ase.calculators.espresso.Espresso` to raise `CalculationFailed` on the first attempt, and succeed on the second.
2.  Run the pipeline.
3.  Check logs for "Calculation failed. Retrying with adjusted parameters...".
4.  Verify the output file is generated correctly.

## 2. Behavior Definitions

### Feature: DFT Self-Healing
**Scenario**: Recovering from SCF Failure
  **Given** a structure that causes an SCF convergence error in QE
  **When** the Oracle attempts to calculate its properties
  **Then** the system should detect the failure
  **And** it should automatically reduce the mixing beta parameter
  **And** retry the calculation
  **And** if the second attempt succeeds, return the valid data
  **And** if all retries fail, mark the structure as "invalid" and proceed to the next one
