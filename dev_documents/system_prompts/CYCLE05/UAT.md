# Cycle 05 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 5.1: Hybrid Potential Generation
*   **Priority**: Critical
*   **Goal**: Verify physical safety configuration.
*   **Action**:
    1.  Provide a potential for "Ti" and "O".
    2.  Invoke `input_builder.build(atoms, potential)`.
*   **Expectation**:
    *   Generated script contains `pair_style hybrid/overlay pace zbl`.
    *   Contains `pair_coeff * * zbl 22 8` (Atomic numbers for Ti, O).
    *   Contains `pair_coeff * * pace potential.yace Ti O`.

### Scenario 5.2: Uncertainty Watchdog Trigger (Mock)
*   **Priority**: High
*   **Goal**: Verify Active Learning loop trigger.
*   **Action**:
    1.  Run `DynamicsEngine` with `mock_halt=True` (simulated failure).
    2.  Check the return value.
*   **Expectation**:
    *   Status is `HALTED`.
    *   `structure` property contains an `Atoms` object (the failing snapshot).
    *   Log file mentions "Watchdog triggered".

### Scenario 5.3: Normal Completion (Mock)
*   **Priority**: Medium
*   **Goal**: Verify successful simulation.
*   **Action**:
    1.  Run `DynamicsEngine` with `mock_halt=False`.
    2.  Check the return value.
*   **Expectation**:
    *   Status is `CONVERGED` (or `COMPLETED`).
    *   `trajectory` property contains list of snapshots.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Dynamics Simulation

  Scenario: LAMMPS runs with safety overlay
    Given a trained potential
    When the input script is generated
    Then it must include "pair_style hybrid/overlay"
    And it must include "fix halt" for uncertainty monitoring

  Scenario: Simulation halts on uncertainty
    Given a running MD simulation
    When the extrapolation grade exceeds "5.0"
    Then the simulation should stop immediately
    And the current configuration should be saved for analysis
```
