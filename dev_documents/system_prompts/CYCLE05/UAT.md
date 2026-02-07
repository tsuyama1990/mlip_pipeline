# Cycle 05 UAT: Beyond Nanoseconds

## 1. Test Scenarios

### 1.1. Scenario: Adaptive Policy Switching
**ID**: UAT-CY05-001
**Priority**: Medium
**Description**: Verify that the system changes its exploration strategy based on the learning progress.

**Steps:**
1.  **Setup**: Configure `PolicyEngine` with a rule: "Switch to kMC after 3 iterations".
2.  **Execution**: Run `Orchestrator` (Mock mode).
3.  **Observation**:
    *   Iter 1-3: Logs show "Running Dynamics: LAMMPS MD".
    *   Iter 4: Logs show "Running Dynamics: EON kMC".
4.  **Verification**: The `active_learning` directories should contain `lammps_run` folders for 1-3 and `eon_run` for 4.

### 1.2. Scenario: EON Driver & Watchdog
**ID**: UAT-CY05-002
**Priority**: High
**Description**: Verify that the EON driver calculates forces correctly and halts on uncertainty.

**Steps:**
1.  **Setup**: Prepare a `pace_driver.py` and a valid potential.
2.  **Execution A (Normal)**: Feed a known structure (Si bulk) via stdin. Check stdout for correct energy.
3.  **Execution B (Halt)**: Feed a "garbage" structure (high uncertainty).
4.  **Observation**: The script should exit with a non-zero code (e.g., 100) and print "Halt triggered" to stderr.

### 1.3. Scenario: Finding a Saddle Point (Simulation)
**ID**: UAT-CY05-003
**Priority**: Low (Requires EON installed)
**Description**: Use kMC to find a diffusion event.

**Steps:**
1.  **Setup**: Create a vacancy in a small Al cluster.
2.  **Execution**: Run the `EONClient` wrapper.
3.  **Verification**:
    *   EON should run `process_search`.
    *   It should identify a saddle point (atom jumping into vacancy).
    *   The `product.con` file should show the atom in the new position.

## 2. Behaviour Definitions

**Feature**: Long-Timescale Exploration (kMC)

**Scenario**: Bridging the Gap

**GIVEN** a system trapped in a deep energy basin (e.g., solid at low temp)
**AND** standard MD cannot escape within 1 ns
**WHEN** the Policy Engine selects "kMC" strategy
**THEN** the system should invoke EON to search for saddle points
**AND** the system should find a transition to a new basin (state)
**AND** the simulation time clock should advance by the appropriate amount (e.g., microseconds)

**Feature**: Driver Watchdog

**Scenario**: Safety during Saddle Search

**GIVEN** EON is dragging an atom over a high-energy barrier
**WHEN** the configuration enters a region of high uncertainty ($\gamma > 5.0$)
**THEN** the potential driver script should abort
**AND** EON should stop
**AND** the Orchestrator should capture this high-energy transition state for labelling
