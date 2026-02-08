# Cycle 05 UAT: Dynamics Engine (MD/kMC) & OTF

## 1. Test Scenarios

These scenarios verify the physics simulation and uncertainty monitoring.

### Scenario 05-01: "Hybrid Potential Setup"
**Priority:** P1 (High)
**Description:** Verify that the system generates a valid LAMMPS input script that overlays the physical baseline (ZBL/LJ) with the ACE potential.
**Success Criteria:**
-   **Config:** `dynamics: lammps`, `reference_potential: zbl`.
-   **Action:** Generate input script.
-   **Check:** The script must contain:
    ```lammps
    pair_style hybrid/overlay pace zbl 1.0 2.0
    pair_coeff * * pace potential.yace ...
    pair_coeff * * zbl ...
    ```
-   **Physics Check:** No crash when two atoms are forcibly placed 0.5 Å apart (nuclear fusion regime).

### Scenario 05-02: "On-the-Fly (OTF) Halt"
**Priority:** P1 (High)
**Description:** Verify that the MD engine halts immediately when the uncertainty threshold is breached.
**Success Criteria:**
-   **Config:** `uncertainty_threshold: 5.0`.
-   **Mock:** Simulate a trajectory where $\gamma$ ramps from 1.0 to 10.0.
-   **Action:** Run MD.
-   **Result:** The simulation must stop at the exact timestep where $\gamma > 5.0$.
-   **Output:** The system must return the specific structure that triggered the halt.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Dynamics & OTF

  Scenario: Hybrid Potential Generation
    Given a configuration with ZBL baseline enabled
    When I generate the LAMMPS input script
    Then the "pair_style" should be "hybrid/overlay"
    And "zbl" should be one of the components

  Scenario: Uncertainty Watchdog
    Given an MD simulation running with max_gamma tracking
    When the max_gamma exceeds the threshold of 5.0
    Then the simulation should halt immediately
    And the high-uncertainty structure should be flagged for learning
```
