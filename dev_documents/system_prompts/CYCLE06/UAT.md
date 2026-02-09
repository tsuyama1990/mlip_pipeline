# Cycle 06 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

These tests verify the core Active Learning Loop.

### Scenario 6.1: OTF Halt Detection
**Objective**: Ensure the system halts when uncertainty is high.
**Priority**: Critical (P0)

*   **Setup**:
    *   Configuration with `dynamics.otf.enabled=True`, `threshold=5.0`.
    *   A structure with known high extrapolation grade (e.g., heavily distorted).
*   **Action**: Run `dynamics.otf_explore(structure, potential)`.
*   **Expected Outcome**:
    *   LAMMPS halts *before* reaching max steps.
    *   `OTFResult.halted` is True.
    *   `OTFResult.halt_structure` corresponds to the last frame.
    *   Log file contains "Fix halt condition met".

### Scenario 6.2: Local Candidate Generation
**Objective**: Verify that the generator creates diverse candidates around a halt structure.
**Priority**: High (P1)

*   **Setup**: A specific "Halt Structure" (e.g., a saddle point configuration).
*   **Action**: Call `generator.generate_local(structure, n=10)`.
*   **Expected Outcome**:
    *   Returns 10 structures.
    *   Structures are slight perturbations of the input (RMSD < 0.5 Å).
    *   Structures span different directions (check displacement vectors).

### Scenario 6.3: Full Active Learning Loop (Mock)
**Objective**: Verify the complete loop logic without expensive DFT.
**Priority**: Medium (P2) - Integration Test.

*   **Setup**:
    *   Mock Oracle (returns random forces).
    *   Mock Trainer (updates potential version).
    *   Dynamics configured to halt every 10 steps (simulated).
*   **Action**: Run `orchestrator.run()`.
*   **Expected Outcome**:
    *   Loop runs for `max_cycles` (e.g., 5).
    *   Dataset size increases in each cycle.
    *   New potential files (v1, v2, v3...) are created.
    *   The loop terminates gracefully.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Active Learning Loop

  Scenario: Halting on high uncertainty
    Given a running MD simulation
    And an uncertainty threshold of 5.0
    When the maximum extrapolation grade exceeds 5.0
    Then the simulation should stop immediately
    And the current structure should be saved as a "Halt Structure"

  Scenario: Recovering from a halt
    Given a simulation halted due to uncertainty
    When the Orchestrator processes the halt event
    Then local candidate structures should be generated around the halt point
    And these candidates should be sent to the Oracle for labeling
    And the potential should be retrained with the new data
    And the simulation should resume from the halt point
```
