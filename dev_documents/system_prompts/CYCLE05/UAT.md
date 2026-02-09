# Cycle 05: User Acceptance Test (UAT) Plan

## 1. Test Scenarios

### Scenario 5.1: Basic NPT Molecular Dynamics (Mock)
**Priority**: High
**Description**: Verify that the system can setup and "run" a basic molecular dynamics simulation using LAMMPS commands.

**Jupyter Notebook**: `tutorials/04_dynamics_test.ipynb`
1.  Initialize `LAMMPSDynamics` with `temperature=300K` and `steps=1000`.
2.  Prepare a dummy `potential.yace` file.
3.  Prepare an initial `Structure` (Si crystal).
4.  Mock the `subprocess.run` call to LAMMPS to return success.
5.  Call `dynamics.explore(potential, structure)`.
6.  Inspect the generated `in.lammps` file.
7.  Assert that `fix npt` command is present with `temp 300.0 300.0`.
8.  Assert that `timestep` is set correctly.

### Scenario 5.2: Hybrid Potential Generation
**Priority**: Critical
**Description**: Verify that the system correctly generates the `pair_style hybrid/overlay` commands to combine ACE with a ZBL baseline. This is crucial for simulation stability.

**Jupyter Notebook**: `tutorials/04_dynamics_test.ipynb`
1.  Create a system with Fe and Pt atoms.
2.  Call the internal method `dynamics._generate_potential_commands()`.
3.  Assert that the output string contains:
    *   `pair_style hybrid/overlay pace zbl ...`
    *   `pair_coeff * * pace potential.yace Fe Pt`
    *   `pair_coeff * * zbl 26 78` (Atomic numbers for Fe and Pt).

### Scenario 5.3: Trajectory Output Handling
**Priority**: Medium
**Description**: Verify that the simulation produces a trajectory file and that the final structure can be parsed from it.

**Jupyter Notebook**: `tutorials/04_dynamics_test.ipynb`
1.  (Requires a real short run or a pre-existing dump file).
2.  Create a dummy LAMMPS dump file with 2 frames.
3.  Call `dynamics._parse_trajectory("dump.lammps")`.
4.  Assert that it returns a list of `Structure` objects.
5.  Assert that the last structure corresponds to the final frame.

## 2. Behavior Definitions

### MD Configuration
**GIVEN** a configuration specifying `ensemble: "npt"`
**WHEN** the input script is generated
**THEN** it should include `fix ... npt ...` command
**AND** set the pressure damping parameter correctly (usually 1000*dt).

### Hybrid Potential Safety
**GIVEN** a potential file `my_pot.yace`
**WHEN** `generate_pair_style` is called
**THEN** it must always include a baseline potential (ZBL or LJ) overlaid on top of PACE
**UNLESS** explicitly configured otherwise (which should be discouraged).

### Execution Robustness
**GIVEN** a LAMMPS execution that fails (non-zero exit code)
**WHEN** `explore` is running
**THEN** it should catch the error
**AND** raise a specific `DynamicsError` containing the last few lines of the log file for debugging.
