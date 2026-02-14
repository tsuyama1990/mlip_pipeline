# Cycle 05 UAT: Dynamics Engine & On-the-Fly Learning

## 1. Test Scenarios

### Scenario 01: Hybrid Potential Config Generation
**Priority**: High
**Description**: Verify that `potential.py` correctly generates LAMMPS commands for a hybrid potential (ACE + ZBL).
**Steps**:
1.  Create a python script `test_hybrid.py`.
2.  Instantiate `DynamicsConfig` with `hybrid_baseline="zbl"`.
3.  Generate LAMMPS commands for a structure (e.g., Fe-Pt).
4.  Check for `pair_style hybrid/overlay pace zbl`.
5.  Check for `pair_coeff * * pace ...` and `pair_coeff * * zbl ...`.
**Expected Result**:
-   Output string matches expected LAMMPS syntax.

### Scenario 02: Detect Halt via LAMMPS Log (Mock)
**Priority**: Critical
**Description**: Verify that `MDInterface` correctly identifies a halt event from a mocked LAMMPS run.
**Steps**:
1.  Create a python script `test_halt.py`.
2.  Create a dummy `log.lammps` file containing `Fix halt condition met`.
3.  Mock `subprocess.run` to return exit code 1 (or whatever `fix halt` produces).
4.  Run `MDInterface.run()`.
**Expected Result**:
-   Returns `HaltInfo(halted=True, step=1234, max_gamma=5.5)`.

### Scenario 03: Extract Bad Structure from Dump
**Priority**: High
**Description**: Verify that `MDInterface` can read a dump file and return the last frame as `ase.Atoms`.
**Steps**:
1.  Create a python script `test_dump.py`.
2.  Create a `dump.lammps` file with 2 frames.
3.  Call `MDInterface.extract_bad_structure(dump_file)`.
**Expected Result**:
-   Returns `ase.Atoms` corresponding to the 2nd frame.
-   Check positions/symbols match the dump file.

### Scenario 04: Active Learning Loop Integration
**Priority**: Critical
**Description**: Verify that the `Orchestrator` correctly sequences the modules when a halt occurs.
**Steps**:
1.  Create a python script `test_orchestrator.py`.
2.  Mock `Dynamics.run` to return `HaltInfo(halted=True)`.
3.  Mock `Oracle.compute` and `Trainer.train`.
4.  Run `Orchestrator.run_cycle()`.
**Expected Result**:
-   `Oracle.compute` was called with the halted structure.
-   `Trainer.train` was called with the new dataset.
-   Current potential updated.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Dynamics & On-the-Fly Learning

  Scenario: Generate Hybrid Potential Commands
    Given a Dynamics configuration with "zbl" baseline
    When I request LAMMPS commands
    Then the output should contain "pair_style hybrid/overlay pace zbl"
    And the ZBL pair coefficients should be set correctly

  Scenario: Detect Simulation Halt
    Given a running MD simulation
    When the extrapolation grade exceeds the threshold
    Then the simulation should halt
    And the system should report the halt step and max gamma

  Scenario: Orchestrate Active Learning
    Given a simulation that halted due to high uncertainty
    When the Orchestrator processes the halt event
    Then it should extract the bad structure
    And send it to the Oracle for labeling
    And retrain the potential with the new data
```
