# Cycle 05 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Basic MD Simulation
**Priority**: High
**Goal**: Verify that MD runs without errors.
**Procedure**:
1.  Configure `dynamics: ensemble: nvt, temperature: 300, steps: 100`.
2.  Provide a trained potential (or mock).
3.  Run the Dynamics Engine.
**Expected Result**:
*   An `in.lammps` file is generated.
*   LAMMPS runs and exits with code 0.
*   A `trajectory.lammps` file is created.

### Scenario 2: Hybrid Potential Verification
**Priority**: High
**Goal**: Verify that the safety baseline is applied.
**Procedure**:
1.  Configure `physics_baseline: zbl`.
2.  Run the Dynamics Engine.
3.  Inspect `in.lammps`.
**Expected Result**:
*   The file contains `pair_style hybrid/overlay`.
*   It does NOT contain a standalone `pair_style pace`.

### Scenario 3: Stress Test (High Temperature)
**Priority**: Medium
**Goal**: Verify stability.
**Procedure**:
1.  Configure `temperature: 2000` (above melting point).
2.  Run MD.
**Expected Result**:
*   The simulation finishes (does not segfault).
*   The atoms have moved significantly (melted).

## 2. Behavior Definitions

```gherkin
Feature: Molecular Dynamics

  Scenario: Running a stable NPT simulation
    GIVEN a crystal structure "MgO"
    AND a trained potential with ZBL baseline
    WHEN "explore" is executed with "T=300K, P=1bar"
    THEN a LAMMPS input script with "fix npt" and "pair_style hybrid/overlay" should be written
    AND the simulation should complete successfully
    AND the volume of the cell should fluctuate around a mean value
```
