# Cycle 05 User Acceptance Testing (UAT)

## 1. Test Scenarios

### SCENARIO 01: LAMMPS Input Generation
**Priority**: High
**Goal**: Verify that the generated `in.lammps` file correctly configures the hybrid potential and uncertainty watchdog.

**Steps**:
1.  Configure `DynamicsConfig`:
    ```yaml
    dynamics:
      type: lammps
      temperature: 300
      steps: 1000
      uncertainty_threshold: 5.0
    ```
2.  Run CLI: `mlip-pipeline explore --config config.yaml --dry-run`.
3.  Check generated `in.lammps`.
4.  Assert `pair_style hybrid/overlay pace zbl` exists.
5.  Assert `fix halt ... v_max_gamma > 5.0` exists.

### SCENARIO 02: OTF Halt Simulation
**Priority**: Medium
**Goal**: Verify the system detects a high uncertainty halt and extracts the correct structure.

**Steps**:
1.  **Mock** the LAMMPS execution to return exit code 1 (or specific halt code) and produce a dummy dump file with high gamma atoms.
2.  Run the pipeline.
3.  Check logs for "Simulation Halted due to High Uncertainty".
4.  Verify that a "candidate structure" is extracted and passed to the Oracle (mock).

## 2. Behavior Definitions

### Feature: On-the-Fly Learning Loop
**Scenario**: Exploring Unknown Configurations
  **Given** an MD simulation running with an ACE potential
  **And** the uncertainty threshold is set to 5.0
  **When** the potential encounters a configuration where the extrapolation grade $\gamma$ exceeds 5.0
  **Then** the simulation should be halted immediately (using `fix halt`)
  **And** the configuration causing the halt should be saved
  **And** the system should flag this structure for DFT labeling
