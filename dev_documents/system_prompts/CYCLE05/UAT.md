# Cycle 05 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

These tests verify the Dynamics Engine's ability to run molecular dynamics.

### Scenario 5.1: Basic MD Execution (Smoke Test)
**Objective**: Ensure LAMMPS can be invoked and run a simple simulation.
**Priority**: High (P1)

*   **Setup**: Configuration with `dynamics.engine="lammps"`. A structure of 64 atoms.
*   **Action**: Call `explore(structure, potential, settings={'n_steps': 100})`.
*   **Expected Outcome**:
    *   LAMMPS runs without error.
    *   `log.lammps` contains "Loop time of ...".
    *   A trajectory file (dump.lammps) is created.
    *   The trajectory contains 100+1 frames.

### Scenario 5.2: Hybrid Potential (Core Repulsion Safety)
**Objective**: Verify that the ZBL baseline prevents catastrophic overlaps.
**Priority**: Critical (P0) - Physics correctness.

*   **Setup**:
    *   Two atoms at very close distance (e.g., 0.5 Å).
    *   Potential configured with `baseline="ZBL"`.
*   **Action**: Run 10 steps of NVE dynamics.
*   **Expected Outcome**:
    *   The atoms repel each other strongly.
    *   The distance increases significantly (e.g., > 1.5 Å).
    *   The simulation does not crash with "Atom lost" or "Segmentation fault".
    *   The generated `in.lammps` file clearly shows `pair_style hybrid/overlay pace zbl`.

### Scenario 5.3: Trajectory Validity (Thermostat)
**Objective**: Verify that the NVT thermostat controls the temperature.
**Priority**: Medium (P2)

*   **Setup**: Run NVT at 300K for 1000 steps on a equilibrated system.
*   **Action**: Parse `log.lammps` or the trajectory.
*   **Expected Outcome**:
    *   The average temperature over the last 500 steps is approximately 300K (within fluctuations).
    *   The energy is conserved (drift is minimal).

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Molecular Dynamics Execution

  Scenario: Running a simulation with Hybrid Potential
    Given a potential trained with a ZBL baseline
    And a structure containing "Fe" and "Pt" atoms
    When I configure a LAMMPS simulation
    Then the input file should contain "pair_style hybrid/overlay pace zbl"
    And the input file should map "Fe" and "Pt" to their atomic numbers (26 and 78) for ZBL
    And the simulation should complete successfully
```
