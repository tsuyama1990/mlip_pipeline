# Cycle 05 UAT: Dynamics & OTF Learning

## 1. Test Scenarios

### Scenario 5.1: LAMMPS Input Generation
*   **ID**: S05-01
*   **Priority**: High
*   **Description**: Verify the generated LAMMPS input file is valid and contains safety features.
*   **Steps**:
    1.  Create a `DynamicsConfig` with `temp=300`, `steps=1000`.
    2.  Provide a `Structure` and `Potential` path.
    3.  Call `generate_lammps_input()`.
    4.  Inspect `in.lammps`.
*   **Expected Result**:
    *   File contains `pair_style hybrid/overlay pace zbl`.
    *   File contains `fix halt ... v_max_gamma > 5.0`.
    *   File contains `dump custom ...`.

### Scenario 5.2: OTF Halt Detection (Mocked)
*   **ID**: S05-02
*   **Priority**: Critical
*   **Description**: Verify the system detects a halted simulation and extracts the problematic structure.
*   **Steps**:
    1.  Use `MockLammps` configured to halt at step 500.
    2.  Run `dynamics.explore()`.
    3.  Check the return value.
*   **Expected Result**:
    *   The function returns a list of structures (candidates).
    *   The log indicates "Simulation halted due to high uncertainty".
    *   The extracted structure corresponds to the snapshot at step 500.

### Scenario 5.3: EON Driver Execution (Stub/Mock)
*   **ID**: S05-03
*   **Priority**: Low (for now)
*   **Description**: Verify the EON driver can be initialized and "run" (mocked).
*   **Steps**:
    1.  Initialize `EonDriver`.
    2.  Call `run_kmc()`.
*   **Expected Result**:
    *   If EON is missing, it should raise `RuntimeError` or log a warning (depending on configuration).
    *   If Mocked, it returns a final structure representing a KMC step.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Dynamics Engine

  Scenario: Hybrid Potential Safety
    GIVEN a generated potential "model.yace"
    WHEN I generate the LAMMPS input script
    THEN the script should enable "hybrid/overlay" pair style
    AND it should include a ZBL baseline for core repulsion

  Scenario: On-the-Fly Learning Trigger
    GIVEN a running MD simulation
    WHEN the maximum extrapolation grade (gamma) exceeds 5.0
    THEN the simulation should stop immediately
    AND the system should extract the current atomic configuration for retraining
```
