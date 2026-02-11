# Cycle 05 User Acceptance Test (UAT)

## 1. Test Scenarios

### Scenario 1: LAMMPS Input Generation
**Priority**: P0 (Critical)
**Description**: Verify that the generated LAMMPS input script correctly implements the Hybrid Potential (ACE+ZBL) and Uncertainty Watchdog.
**Steps**:
1.  Create a `Structure` with elements [Fe, Pt].
2.  Invoke `InputGenerator.generate(structure, config)`.
3.  Inspect the output string.
4.  Verify presence of `pair_style hybrid/overlay pace zbl`.
5.  Verify `pair_coeff` lines include ZBL parameters for Fe (26) and Pt (78).
6.  Verify `fix halt` command uses the correct variable (`v_max_gamma`).

### Scenario 2: Mock Dynamics Execution (Converged)
**Priority**: P1 (High)
**Description**: Verify that the Mock Dynamics engine simulates a successful MD run.
**Steps**:
1.  Configure `config.yaml` with `dynamics.type: mock_converge`.
2.  Run `mlip-runner explore config.yaml`.
3.  Inspect the logs.
4.  Expected: "Simulation completed successfully (1000 steps)."
5.  Check for output trajectory file (`dump.lammps`).

### Scenario 3: Mock Dynamics Execution (Halt)
**Priority**: P1 (High)
**Description**: Verify that the Mock Dynamics engine simulates an uncertainty-driven halt.
**Steps**:
1.  Configure `config.yaml` with `dynamics.type: mock_halt`.
2.  Run the exploration command.
3.  Inspect the logs.
    -   Expected: "Watchdog triggered! Max gamma exceeded threshold at step 50."
4.  Check for the "Halt Structure" (`halt_structure.xyz`). This structure should have high uncertainty tags (mocked).

## 2. Behavior Definitions (Gherkin)

### Feature: Dynamics Simulation

**Scenario**: Generate Hybrid Potential Input
    **Given** a structure with element "Si" (Z=14)
    **When** the Input Generator creates the script
    **Then** the script should contain `pair_coeff * * zbl 14 14`
    **And** it should contain `pair_coeff * * pace potential.yace Si`

**Scenario**: Watchdog Halt Event
    **Given** a running MD simulation
    **When** the extrapolation grade `gamma` exceeds 5.0
    **Then** the simulation should stop immediately (Halt)
    **And** the last frame should be saved as a candidate for active learning
