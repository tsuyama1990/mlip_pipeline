# Cycle 07 User Acceptance Test (UAT)

## 1. Test Scenarios

### Scenario 1: EON kMC Execution (Mock)
**Priority**: P0 (Critical)
**Description**: Verify that the EON Driver successfully launches a kMC search and returns a new (lower energy) structure.
**Steps**:
1.  Configure `config.yaml` with `dynamics.type: mock_eon`.
2.  Run `mlip-runner dynamics config.yaml`.
3.  Inspect the output.
    -   Expected: "EON search completed. Found saddle point."
    -   Expected: "New structure energy < initial energy."

### Scenario 2: Deposition Simulation (Mock)
**Priority**: P1 (High)
**Description**: Verify that the Deposition Module correctly adds atoms over time.
**Steps**:
1.  Configure `config.yaml` with `dynamics.deposition.enabled: true`.
2.  Set `total_atoms: 10`.
3.  Run the simulation.
4.  Inspect the final structure file.
5.  Verify that `N_final = N_initial + 10`.

### Scenario 3: EON Halt Event
**Priority**: P2 (Medium)
**Description**: Verify that if the EON driver detects high uncertainty during a saddle search, it halts the process.
**Steps**:
1.  Configure `config.yaml` with `dynamics.type: mock_eon_halt`.
2.  Run the dynamics.
3.  Inspect the logs.
    -   Expected: "EON halted due to uncertainty."
4.  Verify a `halt_structure.con` is saved.

## 2. Behavior Definitions (Gherkin)

### Feature: Advanced Dynamics

**Scenario**: Adaptive Kinetic Monte Carlo
    **Given** a relaxed structure at a local minimum
    **When** the EON Driver runs a process search
    **Then** it should return a new structure corresponding to a neighboring minimum (basin hopping)
    **And** the energy should be lower or equal (with some probability)

**Scenario**: Atom Deposition
    **Given** an empty substrate surface
    **When** the Deposition Module runs for 1000 steps with rate 0.01
    **Then** 10 new atoms should appear on the surface
    **And** their positions should be physically reasonable (not overlapping) due to the Hybrid Potential
