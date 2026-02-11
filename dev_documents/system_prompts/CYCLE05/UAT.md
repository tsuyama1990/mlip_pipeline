# Cycle 05 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario 5.1: Hybrid Potential Setup
*   **Goal**: Verify that the input script correctly configures a Hybrid potential (ACE + ZBL) to prevent crashes.
*   **Action**:
    1.  User runs `pyacemaker run-loop` with `dynamics.hybrid: true`.
    2.  User inspects the generated `active_learning/iter_XXX/md_run/in.lammps`.
*   **Success Criteria**:
    *   The file contains `pair_style hybrid/overlay`.
    *   It defines two `pair_coeff` commands: one for `pace` (with `.yace` file) and one for `zbl` (or `lj`).
    *   The ZBL coefficients match the atomic numbers of the species (e.g., 26 for Fe).

### Scenario 5.2: Uncertainty Monitoring (Watchdog)
*   **Goal**: Verify that the `fix halt` command is correctly generated to monitor `gamma`.
*   **Action**:
    1.  User inspects the `in.lammps` file again.
    2.  User looks for `compute ... pace ... gamma_mode=1`.
    3.  User looks for `fix ... halt ... v_max_gamma > 5.0`.
*   **Success Criteria**:
    *   The commands are present.
    *   The threshold matches the config value.
    *   The error handling strategy (`error hard` or `soft`) is set.

### Scenario 5.3: Halt Detection (Mock)
*   **Goal**: Verify the system correctly identifies a simulation that was halted due to high uncertainty.
*   **Action**:
    1.  User configures `dynamics.type: mock` with `mock.halt_prob: 1.0`.
    2.  User runs the loop.
    3.  User checks the Orchestrator log.
*   **Success Criteria**:
    *   The log says "Simulation halted at step X due to high uncertainty (gamma=Y)".
    *   The system proceeds to the "Diagnose" phase (which will be fully implemented in Cycle 06).

## 2. Behavior Definitions (Gherkin Style)

### Feature: Hybrid Potential
**Scenario**: Generating LAMMPS input with ZBL baseline
  **Given** a configuration enabling hybrid potential
  **When** the DynamicsEngine generates the input file
  **Then** it should include `pair_style hybrid/overlay`
  **And** it should define coefficients for both ACE and ZBL

### Feature: Uncertainty Watchdog
**Scenario**: Monitoring extrapolation grade
  **Given** a threshold for gamma
  **When** the simulation runs
  **Then** LAMMPS should compute the max gamma every N steps
  **And** it should halt if the value exceeds the threshold

### Feature: Halt Handling
**Scenario**: Parsing a halted run
  **Given** a LAMMPS log file ending with "ERROR: Variable max_gamma > 5.0"
  **When** the Parser reads the file
  **Then** it should return a DynamicsResult with `halted=True`
  **And** it should extract the step number and max gamma value
