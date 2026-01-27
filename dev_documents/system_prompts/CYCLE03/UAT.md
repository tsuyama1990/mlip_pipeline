# Cycle 03: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 01: Hybrid Potential Input Generation
*   **ID**: UAT-CY03-01
*   **Priority**: High
*   **Description**: Verify that the system generates the correct LAMMPS input commands to overlay a ZBL baseline on top of the ACE potential. This is critical for physical robustness.
*   **Pre-conditions**: A `structure.xyz` and a dummy `potential.yace` exist.
*   **Steps**:
    1.  Run `mlip-auto md-dry-run --structure structure.xyz --potential potential.yace --config md_config.yaml`.
    2.  Inspect the generated `in.lammps` file.
*   **Expected Result**:
    *   The file contains `pair_style hybrid/overlay`.
    *   It contains `pair_coeff * * pace ...`.
    *   It contains `pair_coeff * * zbl ...` (or lj/cut).
    *   It does NOT contain a standalone `pair_style pace`.

### Scenario 02: Watchdog Triggering (Simulated)
*   **ID**: UAT-CY03-02
*   **Priority**: Critical
*   **Description**: Verify that the system correctly detects when a simulation is halted due to high uncertainty.
*   **Pre-conditions**: A mock or real LAMMPS executable.
*   **Steps**:
    1.  (Setup) We use a mock script `lmp_mock` that prints "Fix halt condition met" to stdout and exits with code 1.
    2.  Run `mlip-auto md-run --executable ./lmp_mock ...`.
    3.  Check the command output.
*   **Expected Result**:
    *   The system does not throw a Python traceback/error.
    *   The system reports "Simulation Halted: Uncertainty Threshold Exceeded".
    *   The returned status object indicates `halted=True`.

### Scenario 03: Normal MD Completion
*   **ID**: UAT-CY03-03
*   **Priority**: Medium
*   **Description**: Verify that a stable simulation completes successfully and produces a trajectory.
*   **Steps**:
    1.  (Setup) Mock script exits with code 0 and writes "Loop time of ..." to stdout.
    2.  Run `mlip-auto md-run ...`.
*   **Expected Result**:
    *   System reports "Simulation Completed".
    *   The trajectory file `dump.lammps` is preserved.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Dynamics Engine with Uncertainty Monitoring

  Scenario: Generate Robust Input
    GIVEN an input structure containing Titanium and Oxygen
    WHEN I request an MD input file
    THEN the 'pair_style' should be 'hybrid/overlay'
    AND 'zbl' parameters for Ti-Ti, Ti-O, and O-O should be defined

  Scenario: Detect High Uncertainty Halt
    GIVEN an MD simulation running with 'fix halt'
    WHEN the max_gamma value exceeds the threshold of 5.0
    AND LAMMPS terminates with a non-zero exit code
    THEN the LammpsRunner should catch the error
    AND the LogParser should identify "Fix halt" in the logs
    AND the return status should be 'Halted' (not 'Failed')
```
