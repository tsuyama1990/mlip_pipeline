# Cycle 04: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Hybrid Potential Input Check
**Priority**: High
**Goal**: Ensure LAMMPS input is safe (contains ZBL).
**Procedure**:
1.  Generate `in.lammps` via `DynamicsPhase`.
2.  Read the file.
3.  Check for `pair_style hybrid/overlay`.
**Success Criteria**:
*   File contains `hybrid/overlay`.
*   File contains `fix halt`.

### Scenario 2: Uncertainty Halt Detection
**Priority**: High
**Goal**: Verify that the system handles "New Physics" detection correctly.
**Procedure**:
1.  Run `DynamicsPhase` with a Mock LAMMPS that simulates a high-gamma event.
2.  Check the return object of the phase.
**Success Criteria**:
*   Status is `HALTED`.
*   Reason is `Uncertainty`.
*   Dump file path is provided.

## 2. Behavior Definitions

```gherkin
Feature: Dynamics Engine

  Scenario: Generate Safe Input
    GIVEN a set of elements (e.g., Al, Cu)
    WHEN the InputWriter generates the script
    THEN it should include "pair_style hybrid/overlay"
    AND it should define ZBL parameters for Al-Al, Al-Cu, Cu-Cu

  Scenario: Handle Uncertainty Halt
    GIVEN a simulation running in exploration mode
    WHEN the extrapolation grade exceeds the threshold
    THEN the simulation should stop immediately
    AND the system should flag the last frame as a candidate
```
