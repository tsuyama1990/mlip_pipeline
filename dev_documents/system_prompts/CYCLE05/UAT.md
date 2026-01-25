# Cycle 05 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 5.1: Uncertainty Detection (Watchdog)
*   **Priority:** Critical
*   **Description:** The system correctly stops the simulation when the extrapolation grade exceeds the user-defined threshold.
*   **Input:** Simulation with `uncertainty_threshold = 5.0`.
*   **Simulation:** Use a mock potential or configuration that guarantees high gamma values.
*   **Expected Output:**
    *   Simulation stops before `max_steps`.
    *   Runner returns status `HALTED`.
    *   Log file contains "Fix halt condition met".

### Scenario 5.2: Periodic Embedding
*   **Priority:** High
*   **Description:** The system extracts a valid, periodic structure suitable for DFT from a halted MD snapshot.
*   **Input:** Large MD snapshot (1000 atoms) with a local high-uncertainty region.
*   **Expected Output:**
    *   A smaller `Atoms` object (e.g., 64-100 atoms).
    *   `pbc` is `[True, True, True]`.
    *   The high-uncertainty atoms are centered in the new cell.

### Scenario 5.3: Recovery Loop
*   **Priority:** Medium
*   **Description:** The Orchestrator correctly transitions from `HALTED` to `SELECTION` state.
*   **Input:** Workflow in `EXPLORATION` state.
*   **Event:** Runner returns `HALTED`.
*   **Expected Output:**
    *   Workflow state updates to `SELECTION`.
    *   The extraction and embedding process is triggered.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Active Learning Interrupt

  Scenario: Halt on High Uncertainty
    Given an MD simulation configured with uncertainty threshold 2.0
    When the potential reports a gamma value of 2.1
    Then LAMMPS should terminate immediately
    And the system should record a "HALT" event
    And the last structure should be saved

  Scenario: Prepare DFT Structure
    Given a halted structure with 1000 atoms
    When I apply periodic embedding
    Then I should get a smaller structure (approx 100 atoms)
    And the structure should be fully periodic
    And the structure should contain the atoms that triggered the halt
```
