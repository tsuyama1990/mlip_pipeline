# Cycle 06 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Triggering a Halt (Simulated)
**Priority**: High
**Goal**: Verify watchdog activation.
**Procedure**:
1.  Configure `uncertainty_threshold: 0.0` (Force immediate halt).
2.  Run Dynamics.
**Expected Result**:
*   LAMMPS exits almost immediately.
*   Orchestrator logs "Simulation halted due to high uncertainty".
*   A structure file `halted_structure.xyz` is saved.

### Scenario 2: Local Candidate Generation
**Priority**: Medium
**Goal**: Verify candidate cloud.
**Procedure**:
1.  Manually trigger the `LocalCandidateGenerator` on a `halted_structure.xyz`.
2.  Inspect output.
**Expected Result**:
*   A list of 10-20 structures is produced.
*   They are slight variations of the input (not identical).

### Scenario 3: Full OTF Loop (Mocked)
**Priority**: Critical
**Goal**: Verify the "Self-Healing" cycle.
**Procedure**:
1.  Set up a mock environment where the potential is initially "bad" (always triggers halt).
2.  After 1 retraining, the potential becomes "good" (mocked).
3.  Run the Orchestrator.
**Expected Result**:
*   Step 1: Dynamics runs -> Halts.
*   Step 2: Retraining occurs.
*   Step 3: Dynamics resumes -> Completes successfully.

## 2. Behavior Definitions

```gherkin
Feature: On-the-Fly Learning

  Scenario: Detecting unknown physics
    GIVEN a simulation running with a Pacemaker potential
    WHEN the extrapolation grade (gamma) exceeds 5.0
    THEN the simulation should stop
    AND the Python driver should catch the "Halt" signal
    AND the problematic structure should be extracted for labeling

  Scenario: Healing the potential
    GIVEN a halted structure
    WHEN the "Local Candidate Generator" creates perturbations
    AND these are labeled by DFT
    AND the potential is retrained
    THEN the new potential should have a lower gamma for that structure
```
