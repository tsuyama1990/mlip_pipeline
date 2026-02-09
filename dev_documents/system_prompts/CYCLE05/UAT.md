# Cycle 05 UAT: Dynamics (LAMMPS)

## 1. Test Scenarios

### Scenario 5.1: Hybrid Potential Configuration
*   **Goal:** Verify that the LAMMPS input script correctly overlays ZBL on the MLIP potential.
*   **Steps:**
    1.  Provide a `Structure` with Ti and O atoms.
    2.  Run `Dynamics.explore()` with `config.dynamics.baseline="ZBL"`.
    3.  Inspect the generated `in.lammps`.
*   **Expected Behavior:**
    *   The file contains `pair_style hybrid/overlay pace zbl 1.0 2.0`.
    *   It contains `pair_coeff * * zbl 22 8` (Ti=22, O=8).
    *   It contains `pair_coeff * * pace potential.yace Ti O`.

### Scenario 5.2: Basic MD Simulation (NPT)
*   **Goal:** Verify that a standard NPT simulation runs to completion.
*   **Steps:**
    1.  Provide an initial `Structure` (e.g., 200 atom crystal).
    2.  Run `Dynamics.explore(steps=1000, temp=300, press=1.0)`.
    3.  Inspect the output directory.
*   **Expected Behavior:**
    *   `log.lammps` exists and shows thermodynamic evolution.
    *   `dump.lammps` contains 100 frames (dump frequency 10).
    *   The simulation does not crash.
    *   The final temperature is approximately 300K.

### Scenario 5.3: Error Handling
*   **Goal:** Verify that the system handles LAMMPS errors gracefully.
*   **Steps:**
    1.  Simulate a "Lost Atoms" error (e.g., by manually setting bad positions or configuring a faulty thermostat in a test case).
    2.  Run `Dynamics.explore()`.
    3.  Check orchestrator logs.
*   **Expected Behavior:**
    *   The system detects the non-zero exit code or error message in the log.
    *   It raises a `DynamicsError` (or similar).
    *   It saves the state (last known good structure) for debugging.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Molecular Dynamics

  Scenario: Generate hybrid potential input
    Given a system with Titanium and Oxygen
    When I configure the dynamics engine
    Then the input script should enable "hybrid/overlay" pair style
    And the ZBL parameters should match atomic numbers 22 and 8

  Scenario: Execute NPT simulation
    Given an equilibrated crystal structure
    When I run a 1000-step NPT simulation at 300K
    Then the simulation should complete successfully
    And the output trajectory should contain 100 frames
    And the temperature should fluctuate around 300K
```
