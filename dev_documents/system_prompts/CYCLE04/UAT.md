# Cycle 04 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 4.1: Hybrid Potential Configuration
*   **Priority:** Critical
*   **Description:** The system must generate a LAMMPS input file that correctly overlays the ZBL potential on top of the PACE potential. This is a safety requirement.
*   **Input:** System with Al and Cu.
*   **Expected Output:**
    *   `in.lammps` contains `pair_style hybrid/overlay pace zbl ...`.
    *   ZBL coefficients for Al (13) and Cu (29) are correctly set.

### Scenario 4.2: Stable MD Execution (Happy Path)
*   **Priority:** High
*   **Description:** The system runs a short MD simulation (e.g., 100 steps) on a stable crystal structure.
*   **Pre-requisites:** LAMMPS installed (or mocked).
*   **Input:** Valid potential, Crystal structure.
*   **Expected Output:**
    *   Simulation completes with exit code 0.
    *   `log.lammps` shows temperature fluctuating around the target (e.g., 300K).
    *   Final structure is returned.

### Scenario 4.3: Crash Handling
*   **Priority:** Medium
*   **Description:** If LAMMPS crashes (e.g., "Bond atoms missing" or "Segmentation fault"), the system captures the error and raises a Python exception rather than hanging.
*   **Input:** Corrupted potential or extreme structure.
*   **Expected Output:**
    *   `SimulationError` raised.
    *   Error message from LAMMPS stdout is preserved.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: MD Simulation

  Scenario: Generate Hybrid Input
    Given a generic binary system (Ti-O)
    When I generate the LAMMPS input
    Then the pair style should be "hybrid/overlay"
    And the ZBL pair coefficients should be "22 8"

  Scenario: Run MD
    Given a valid potential file
    And an initial structure
    When I run a 1000 step NVT simulation
    Then the simulation should finish successfully
    And the final temperature should be close to 300K
```
