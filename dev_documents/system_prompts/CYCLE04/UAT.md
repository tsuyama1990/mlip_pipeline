# Cycle 04 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 4.1: Hybrid Potential Logic
**Goal**: Verify that the system correctly generates hybrid potentials (ACE + ZBL) for core repulsion.
**Priority**: High (P1) - Prevents physical collapse.
**Steps**:
1.  Configure `config.yaml` with `dynamics.type: lammps` and `uncertainty_threshold: 5.0`.
2.  Run the Orchestrator with a small `Structure` (e.g., FeO).
3.  Intercept the generated `in.lammps` file (via logging or pausing).
4.  Inspect the file.
**Success Criteria**:
*   The file contains `pair_style hybrid/overlay pace zbl`.
*   The `pair_coeff` lines for ZBL correctly reference atomic numbers (Fe: 26, O: 8).

### Scenario 4.2: Uncertainty Watchdog (Halt Test)
**Goal**: Verify that the simulation stops when the potential is uncertain.
**Priority**: Critical (P0) - Active Learning depends on this.
**Steps**:
1.  Configure `config.yaml` with `uncertainty_threshold: 0.001` (extremely low).
2.  Run the Orchestrator with a valid `potential.yace`.
3.  Let the simulation run.
**Success Criteria**:
*   The simulation should halt almost immediately (e.g., step 10).
*   The log should report "Dynamics Halted due to High Uncertainty".
*   The `ExplorationResult` should contain the "halt structure" (the problematic configuration).

## 2. Behaviour Definitions (Gherkin)

```gherkin
Feature: Dynamics & Halt Logic

  Scenario: Generate Hybrid Potential Input
    Given a structure with Fe and O atoms
    And a config requesting "lammps" dynamics
    When I generate the LAMMPS input
    Then it should include "pair_style hybrid/overlay pace zbl"
    And it should define ZBL coeffs for Fe-Fe, Fe-O, O-O

  Scenario: Halt on High Uncertainty
    Given a valid potential and structure
    And a dynamics config with uncertainty_threshold=0.0
    When I run the dynamics engine
    Then the simulation should halt within 100 steps
    And the result should be marked as halted
    And the returned structure should be the one causing the halt
```
