# User Acceptance Test (UAT): Cycle 04

## 1. Test Scenarios

### Scenario 04-01: The Safety Net (Priority: High)
**Objective**: Verify that the generated input files correctly implement the Hybrid Potential (ACE + ZBL).

**Description**:
The user is terrified of "holes" in the potential causing their simulation to explode. They want visual confirmation that ZBL is active.

**User Journey**:
1.  User configures a run for Aluminum.
2.  User inspects the generated `active_learning/iter_001/md_run/in.lammps` file.
3.  User sees `pair_style hybrid/overlay pace zbl 1.0 2.0`.
4.  User sees `pair_coeff * * zbl 13 13`.
5.  User runs the simulation (Real Mode) and checks the log.
6.  The log confirms "Pair style: hybrid/overlay".

**Success Criteria**:
*   The `in.lammps` file contains the hybrid commands.
*   The simulation runs without error (if potential exists).

### Scenario 04-02: The Red Flag (Priority: High)
**Objective**: Verify that the system stops the simulation when uncertainty is high.

**Description**:
The user wants to trust the system to learn autonomously. This requires the system to stop *before* it generates nonsense.

**User Journey**:
1.  User runs a simulation with a "weak" potential (Mock Mode: configured to fail).
2.  The system starts MD.
3.  At step 500, the mock LAMMPS reports "Gamma = 6.0" (Threshold 5.0).
4.  The system stops the MD.
5.  The Orchestrator logs "High uncertainty detected at step 500."
6.  The Orchestrator initiates the "Selection" phase.

**Success Criteria**:
*   The `WorkflowState` updates to "Selection".
*   The system does not run for the full 10,000 steps requested.

## 2. Behavior Definitions (Gherkin)

### Feature: Dynamics Engine

```gherkin
Feature: LAMMPS Integration

  Scenario: Hybrid Potential Generation
    GIVEN a configuration for "Ti-O" system
    WHEN the InputGenerator creates the LAMMPS input
    THEN the pair_style should be "hybrid/overlay"
    AND there should be a "zbl" pair coefficient for Ti-Ti, Ti-O, and O-O

  Scenario: OTF Halt Handling
    GIVEN a running MD simulation
    WHEN the extrapolation grade (gamma) exceeds the threshold
    THEN the simulation should terminate with a specific error code
    AND the Orchestrator should capture the termination
    AND the Orchestrator should identify the last valid structure
```
