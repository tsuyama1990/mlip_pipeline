# Cycle 06 UAT: Scalable Inference Engine (Part 1)

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-06-01** | High | **LAMMPS Simulation Launch** | Verify that the system can generate a valid LAMMPS input script, convert the atomic structure to LAMMPS data format, and launch the simulation (mocked). The input script must reference the provided potential file. |
| **UAT-06-02** | High | **Uncertainty Mining** | Verify that if the simulation encounters high-uncertainty configurations (simulated via dump file output), the system correctly identifies and extracts them for future labeling. The extracted structures must contain the `gamma` values. |
| **UAT-06-03** | Medium | **Trajectory Analysis** | Verify that basic thermodynamic properties (Temperature, Pressure, Potential Energy) can be extracted from the LAMMPS log files for post-run analysis. |
| **UAT-06-04** | Low | **Ensemble Control** | Verify that changing the config from NVT to NPT correctly changes the LAMMPS `fix` commands in the generated script. |

### Recommended Demo
Create `demo_06_inference.ipynb`.
1.  **Block 1**: Setup `InferenceConfig` for a 10ps run at 1000K.
2.  **Block 2**: Run `LammpsRunner.run()` (mock mode).
3.  **Block 3**: Show the generated `in.lammps` file. Highlight the `compute gamma` line.
4.  **Block 4**: (Mock) Create a `dump.gamma` file with one frame containing high-gamma atoms.
5.  **Block 5**: Run `UncertaintyChecker`. Show that it loads the frame as an `Atoms` object and prints the max gamma.

## 2. Behavior Definitions

### Scenario: Stable Simulation
**GIVEN** a trained potential and a crystal structure.
**WHEN** the simulation runs in a stable regime (Gamma < 2.0).
**THEN** the `dump.gamma` file (configured to write only if Gamma > 5.0) should be empty or non-existent.
**AND** the log file should show the run completing all steps.
**AND** the system should return status "Converged".

### Scenario: Active Learning Trigger
**GIVEN** a simulation where the temperature is too high.
**WHEN** the extrapolation grade $\gamma$ exceeds 5.0 at step 500.
**THEN** LAMMPS (via `dump_modify`) should write the current snapshot to disk.
**AND** the Python runner, upon completion, should detect this file.
**AND** it should parse the file into an `Atoms` object.
**AND** it should report "Found 1 uncertain configuration".
**AND** the returned object should contain the step number 500 in its info.

### Scenario: Input Validation
**GIVEN** an `InferenceConfig` with `timestep = 0.0`.
**WHEN** the runner is initialized.
**THEN** it should raise a `ValidationError` (Timestep must be positive).
**GIVEN** a potential file path that does not exist.
**WHEN** `run()` is called.
**THEN** it should raise a `FileNotFoundError` before calling subprocess.

### Scenario: NPT vs NVT
**GIVEN** an `InferenceConfig` with `ensemble='npt'` and `pressure=100.0`.
**WHEN** the script is generated.
**THEN** it should contain `fix 1 all npt temp ... iso 100.0 100.0 ...`.
