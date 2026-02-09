# Cycle 06 UAT: OTF Loop

## 1. Test Scenarios

### Scenario 6.1: Active Learning Halt
*   **Goal:** Verify that a high-gamma structure triggers an active learning cycle.
*   **Steps:**
    1.  Create a "bad" initial potential (e.g., poorly trained).
    2.  Run MD.
    3.  Monitor logs for "Halt condition met".
*   **Expected Behavior:**
    *   LAMMPS exits with an error.
    *   Orchestrator catches it.
    *   A new structure is extracted.
    *   The `active_learning` directory contains `candidates` and `dataset`.

### Scenario 6.2: Loop Convergence
*   **Goal:** Verify that the system eventually learns enough to run without halting.
*   **Steps:**
    1.  Start with a "bad" potential.
    2.  Allow the Orchestrator to run for 5 cycles.
    3.  Use a Mock Oracle that returns "perfect" energies (e.g., from an EMT calculator).
*   **Expected Behavior:**
    *   The number of halts decreases over time.
    *   The final cycle runs to completion (e.g., 1000 steps without halt).
    *   The RMSE on a validation set decreases.

### Scenario 6.3: Resume Logic
*   **Goal:** Verify that the simulation can be restarted from the halted checkpoint.
*   **Steps:**
    1.  Run MD, force a halt at step 50.
    2.  Let the loop complete (train new potential).
    3.  The Orchestrator should trigger a restart.
    4.  Inspect the new `log.lammps`.
*   **Expected Behavior:**
    *   The simulation resumes from step 50 (or slightly earlier).
    *   The total steps run is correct (e.g., if target was 100, and halt at 50, it runs 50 more).

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Active Learning Loop

  Scenario: Detect high uncertainty
    Given an MD simulation running with "fix halt"
    When the extrapolation grade exceeds 5.0
    Then the simulation should stop immediately
    And the orchestrator should identify the halted structure

  Scenario: Refine potential
    Given a halted structure
    When the active learning loop processes it
    Then new labeled data should be added to the training set
    And a new potential version should be trained
    And the new potential should have lower uncertainty for that structure

  Scenario: Resume simulation
    Given a simulation that was halted at step 500
    And a refined potential
    When the orchestrator resumes the simulation
    Then it should continue from step 500
    And it should use the new potential file
```
