# Cycle 05: Dynamics UAT

## 1. Test Scenarios

### Scenario 05-01: MD with Halt (OTF)
**Priority**: Critical
**Goal**: Verify Active Learning Logic.
**Description**:
1.  Run NVE MD on a structure far from equilibrium (e.g., compressed by 20%).
2.  Set `halt_threshold=1.0` (low).
3.  The simulation should halt quickly.
**Expected Outcome**:
-   The return code from `Dynamics.run_md` indicates `halted=True`.
-   The last snapshot in `dump.lammps` corresponds to the high-uncertainty configuration.

### Scenario 05-02: Hybrid Potential Safety
**Priority**: High
**Goal**: Verify Core Repulsion.
**Description**:
1.  Create a dimer with $r = 0.5 \AA$ (nuclear fusion distance).
2.  Run strict minimization or single-point energy calculation using `pair_style hybrid/overlay`.
**Expected Outcome**:
-   The energy is massively positive (repulsive) due to ZBL.
-   Without ZBL, the ACE potential might predict a small or negative energy (hole).

### Scenario 05-03: kMC Execution (Mock EON)
**Priority**: Medium
**Goal**: Verify long-timescale logic.
**Description**:
1.  Run `Dynamics.run_kmc` on a simple vacancy diffusion event.
2.  Use a mock EON binary if the real one is missing.
**Expected Outcome**:
-   The system returns a `KMCResult` with transition state energies and barriers.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Dynamics Engine

  Scenario: Run MD on stable Bulk
    Given a well-trained potential
    When I run NVE dynamics for 1000 steps
    Then the simulation should complete without halting
    And energy should be conserved

  Scenario: Run MD on unknown configuration
    Given a potential trained only on Bulk
    When I run NVE dynamics on a Surface structure
    Then the extrapolation grade gamma should exceed threshold
    And the simulation should halt
    And the high-gamma structure should be returned
```

## 3. Jupyter Notebook Validation (`tutorials/04_Dynamics_Test.ipynb`)
-   **Setup**: Load potential from Cycle 04.
-   **Run MD**: `dynamics.run_md(atoms, settings={'steps': 1000})`.
-   **Plot**: Use `matplotlib` to plot Total Energy vs Time.
-   **Visualize**: Use `ovito` (if installed) or `ase.visualize` to see the trajectory.
