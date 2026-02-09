# Cycle 07 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

These tests verify the Advanced Dynamics capabilities, focusing on rare events and surface growth.

### Scenario 7.1: EON Integration (Driver Verification)
**Objective**: Ensure the generated driver script correctly communicates with the EON software.
**Priority**: High (P1)

*   **Setup**: Configuration with `dynamics.eon.enabled=True`.
*   **Action**: Call `eon.run_kmc(structure, potential)`.
*   **Expected Outcome**:
    *   `pace_driver.py` is created in the work directory.
    *   EON executes the driver.
    *   The driver returns valid energies and forces (standard out).
    *   If `gamma > threshold`, the driver exits with a unique error code (e.g., 100).

### Scenario 7.2: Thin Film Deposition (MD)
**Objective**: Verify the ability to grow a film atom-by-atom.
**Priority**: High (P1)

*   **Setup**:
    *   Substrate: MgO (fixed bottom layers).
    *   Deposition Species: Fe, Pt (alternating).
    *   Rate: 1 atom per 1000 steps.
    *   Total Atoms: 10.
*   **Action**: Call `lammps_deposition.run_deposition(substrate, potential)`.
*   **Expected Outcome**:
    *   Trajectory shows 10 atoms appearing above the surface.
    *   Atoms stick to the surface (adhesion).
    *   Temperature remains stable (thermostat works).
    *   Core repulsion prevents fusion.

### Scenario 7.3: Cluster Ordering (aKMC)
**Objective**: Verify that aKMC finds lower energy states over long timescales.
**Priority**: Medium (P2) - Scientific Verification.

*   **Setup**:
    *   Start with a disordered FePt cluster (high energy).
    *   Potential: Trained FePt potential.
*   **Action**: Run aKMC for 100 steps (or until convergence).
*   **Expected Outcome**:
    *   Energy decreases monotonically (or fluctuates around a minimum).
    *   Visual inspection shows increased chemical ordering (e.g., separation of Fe and Pt layers or L10 pattern).
    *   Process search finds saddle points and minima.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Advanced Dynamics Simulation

  Scenario: Simulating thin film growth
    Given a pristine MgO substrate
    And a deposition rate of 1 atom per nanosecond
    When I run the deposition simulation for 10 nanoseconds
    Then 10 new atoms should be added to the system
    And they should form a cluster on the surface

  Scenario: Searching for rare events with EON
    Given a local minimum structure on the Potential Energy Surface
    When I run an adaptive Kinetic Monte Carlo search
    Then the system should explore saddle points
    And the system should transition to a new basin of attraction
    And the time elapsed should be much longer than typical MD timescales
```
