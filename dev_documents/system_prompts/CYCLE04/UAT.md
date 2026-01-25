# Cycle 04 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Hybrid Potential Configuration
*   **ID**: UAT-04-01
*   **Priority**: Critical
*   **Description**: Verify that the generated LAMMPS input correctly sets up the hybrid potential (ACE + ZBL).
*   **Success Criteria**:
    *   Input file contains `pair_style hybrid/overlay pace ... zbl ...`.
    *   Coefficients for both potentials are set correctly for all element pairs.

### Scenario 2: Uncertainty Watchdog Trigger
*   **ID**: UAT-04-02
*   **Priority**: Critical
*   **Description**: Simulate a run where the uncertainty exceeds the threshold.
*   **Success Criteria**:
    *   Mock LAMMPS output contains "Fix halt condition met".
    *   The runner identifies the run as `HALTED` (not `FAILED` or `COMPLETED`).
    *   The system logs "Simulation halted at step X due to high uncertainty".

### Scenario 3: Candidate Extraction
*   **ID**: UAT-04-03
*   **Priority**: High
*   **Description**: Verify that the system can extract the exact structure that caused the halt.
*   **Success Criteria**:
    *   Given a dump file with 100 frames.
    *   Simulate halt at frame 42.
    *   The extracted structure corresponds exactly to frame 42 (check atom positions).

## 2. Behavior Definitions

```gherkin
Feature: Dynamics Engine (LAMMPS)

  As a researcher
  I want the simulation to stop automatically when the physics becomes uncertain
  So that I don't generate garbage data or crash the simulation

  Scenario: Watchdog Activation
    GIVEN a running MD simulation
    AND an uncertainty threshold of 5.0
    WHEN the maximum gamma value reaches 5.1
    THEN LAMMPS should terminate immediately
    AND the runner should report the halt status

  Scenario: Hybrid Potential Safety
    GIVEN a system config with ZBL enabled
    WHEN I generate the LAMMPS input
    THEN the pair style should be "hybrid/overlay"
    AND the ZBL potential should be applied to all pairs
```
