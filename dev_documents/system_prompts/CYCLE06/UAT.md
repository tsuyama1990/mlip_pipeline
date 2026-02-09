# Cycle 06: User Acceptance Test (UAT) Plan

## 1. Test Scenarios

### Scenario 6.1: Watchdog Triggering (Mock)
**Priority**: Critical
**Description**: Verify that the LAMMPS simulation correctly halts when the uncertainty metric exceeds the threshold.

**Jupyter Notebook**: `tutorials/05_otf_loop_test.ipynb`
1.  Initialize `LAMMPSDynamics` with `uncertainty_threshold=5.0`.
2.  Run a simulation where the gamma value is artificially increased (e.g., using a dummy potential or modifying `in.lammps`).
3.  Assert that `result.halted` is True.
4.  Assert that the final structure corresponds to the halt frame.

### Scenario 6.2: OTF Refinement Cycle
**Priority**: Critical
**Description**: Verify that the entire loop (Halt -> Extract -> Embed -> DFT -> Train -> Resume) functions as a cohesive unit.

**Jupyter Notebook**: `tutorials/05_otf_loop_test.ipynb`
1.  Set up the `Orchestrator` with mock components (`MockOracle`, `MockDynamics`).
2.  Configure `MockDynamics` to return `halted=True` on the first call, `halted=False` on the second.
3.  Run `orchestrator.run_cycle(max_cycles=2)`.
4.  Assert that `oracle.compute` was called once.
5.  Assert that `trainer.train` was called once.
6.  Assert that the final state is `CONVERGED`.

### Scenario 6.3: Loop Convergence Behavior
**Priority**: Medium
**Description**: Verify that the system demonstrates improvement over time (longer runs between halts).

**Jupyter Notebook**: `tutorials/05_otf_loop_test.ipynb`
1.  (This is difficult to test deterministically without a real physics engine, but can be mocked).
2.  Mock the `gamma` evolution such that it decreases with each training iteration.
3.  Run the loop for 5 cycles.
4.  Plot the `steps_completed` vs. `cycle_number`.
5.  Assert that `steps_completed` generally increases.

## 2. Behavior Definitions

### Watchdog Logic
**GIVEN** a running MD simulation
**WHEN** the maximum extrapolation grade $\gamma_{max}$ exceeds the configured threshold (e.g., 5.0)
**THEN** the simulation should stop immediately (within the check interval)
**AND** the log file should indicate a "halt" condition (e.g., "Fix halt condition met").

### Structure Extraction
**GIVEN** a halted trajectory file
**WHEN** `extract_high_gamma_structures` is called
**THEN** it should identify the frame where the halt occurred
**AND** extract a cluster centered on the atom with the highest $\gamma$ value.

### Cycle Management
**GIVEN** a halted simulation
**WHEN** the refinement phase completes
**THEN** the new potential should be deployed
**AND** the simulation should restart from the *exact* configuration where it halted (using `read_restart` or similar), or start a new exploration from a similar state.
