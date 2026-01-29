# Cycle 06 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 6.1: MD Simulation with Hybrid Potential
*   **ID**: UAT-06-01
*   **Priority**: High
*   **Description**: MD runs using both ACE and ZBL potentials.
*   **Steps**:
    1.  Configure `DynamicsConfig` with a known potential.
    2.  Run the Dynamics Phase.
    3.  Inspect the generated `in.lammps`.
    4.  Verify it contains `pair_style hybrid/overlay pace ... zbl ...`.

### Scenario 6.2: Uncertainty-Driven Halt
*   **ID**: UAT-06-02
*   **Priority**: High
*   **Description**: The simulation stops when a structure enters a high-uncertainty region.
*   **Steps**:
    1.  Mock the potential to return high gamma values (or use a very low threshold).
    2.  Run the Dynamics Phase.
    3.  Verify the run stops before `md_steps` is reached.
    4.  Verify the logs say "Halted due to high uncertainty".
    5.  Verify a new candidate structure is extracted.

## 2. Behavior Definitions

```gherkin
Feature: Active Learning Dynamics

  Scenario: Normal MD run
    GIVEN a well-trained potential
    WHEN the Dynamics Phase runs
    THEN the simulation should complete without halting
    AND the system should proceed to the next iteration

  Scenario: Encountering unknown physics
    GIVEN a potential with limited training data
    WHEN the MD simulation encounters a rare event (high gamma)
    THEN the "fix halt" trigger should fire
    AND the simulation should stop immediately
    AND the Orchestrator should extract the high-gamma structure for labeling
```
