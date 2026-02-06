# Cycle 05 UAT: The Simulation Run

## 1. Test Scenarios

### Scenario 5: Running MD with Hybrid Potential
**Priority**: High
**Objective**: Verify that the MD engine correctly applies the hybrid potential strategy.

**Steps**:
1.  **Preparation**:
    *   Provide a dummy `potential.yace`.
    *   Config: `explorer.type = "lammps"`.
2.  **Execution**:
    *   Run the exploration step.
3.  **Verification**:
    *   Inspect the generated `in.lammps` file.
    *   **Check**: It must contain `pair_style hybrid/overlay`.
    *   **Check**: It must NOT contain `pair_style pace` alone.
    *   **Check**: The simulation runs (or mock passes) and produces a trajectory file.

## 2. Behavior Definitions

```gherkin
Feature: Hybrid Potential Application

  Scenario: Preventing core overlap
    GIVEN a trained ACE potential
    AND a configuration enabling ZBL baseline
    WHEN the Dynamics Engine generates the LAMMPS input
    THEN it should configure "pair_style hybrid/overlay pace zbl"
    AND it should set ZBL parameters based on the atomic numbers of the species
```
