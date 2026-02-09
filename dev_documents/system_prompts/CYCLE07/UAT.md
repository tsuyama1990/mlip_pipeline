# Cycle 07: User Acceptance Test (UAT) Plan

## 1. Test Scenarios

### Scenario 7.1: EON Simulation Setup
**Priority**: High
**Description**: Verify that the EON directory structure is created correctly with all necessary input files.

**Jupyter Notebook**: `tutorials/06_advanced_dynamics.ipynb`
1.  Initialize `EONDriver` with `temperature=600.0`.
2.  Prepare a dummy `potential.yace` and `Structure`.
3.  Call `driver.setup_simulation(structure, potential)`.
4.  Assert that `reactant.con` exists.
5.  Assert that `config.ini` exists and contains `temperature = 600.0`.
6.  Assert that `pace_driver.py` is present and executable.

### Scenario 7.2: kMC Execution (Mock)
**Priority**: Medium
**Description**: Verify that the system can execute the EON client and parse its output.

**Jupyter Notebook**: `tutorials/06_advanced_dynamics.ipynb`
1.  Mock the `subprocess.run` call to simulate `eonclient` running for 10 steps.
2.  Mock the creation of `processes.dat` with dummy time data.
3.  Call `driver.run_kmc()`.
4.  Assert that `ExplorationResult.meta['time']` is > 0.
5.  Assert that `ExplorationResult.halted` is False.

### Scenario 7.3: On-the-Fly Halt Detection in kMC
**Priority**: Critical
**Description**: Verify that the system correctly handles high-uncertainty events during saddle point searches.

**Jupyter Notebook**: `tutorials/06_advanced_dynamics.ipynb`
1.  Mock the `subprocess.run` call to return exit code 100 (simulating halt).
2.  Call `driver.run_kmc()`.
3.  Assert that `ExplorationResult.halted` is True.
4.  Assert that the driver attempts to locate `bad_structure.con` (or similar output from `pace_driver.py`).

## 2. Behavior Definitions

### EON Configuration
**GIVEN** a config for "dimer" method
**WHEN** `setup_simulation` runs
**THEN** `config.ini` should have `[Process Search]` section with `min_mode_method = dimer`.

### Driver Script Logic
**GIVEN** an atomic configuration with max extrapolation grade $\gamma=10.0$
**WHEN** `pace_driver.py` calculates energy
**THEN** it should write the structure to a file
**AND** exit with code 100
**AND** NOT print energy to stdout (to signal error to EON).

### Orchestrator Handling
**GIVEN** a kMC simulation halted with code 100
**WHEN** the Orchestrator resumes
**THEN** it should treat this exactly like an MD halt
**AND** initiate the "Extract -> Embed -> DFT" refinement loop.
