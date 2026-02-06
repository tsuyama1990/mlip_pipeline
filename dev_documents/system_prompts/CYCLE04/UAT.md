# Cycle 04 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario 4.1: Hybrid Potential Setup
**Priority**: Critical
**Description**: Verify that the generated LAMMPS script correctly implements the Hybrid/Overlay potential for safety.
**Steps**:
1.  Configure the Explorer with `physics_baseline="zbl"`.
2.  Generate the `in.lammps` file.
3.  Inspect the file content.
4.  **Expectation**:
    *   Line: `pair_style hybrid/overlay pace ... zbl ...`
    *   Line: `pair_coeff * * pace ...`
    *   Line: `pair_coeff * * zbl ...`

### Scenario 4.2: Trajectory Parsing
**Priority**: High
**Description**: Verify that the system can read multi-frame trajectories from LAMMPS.
**Steps**:
1.  Provide a sample `dump.lammps` file containing 10 frames of an Fe-Pt system.
2.  Call `explorer.parse_output()`.
3.  **Expectation**:
    *   Return a list of 10 `ase.Atoms` objects.
    *   The chemical symbols (Fe, Pt) should be correctly mapped from LAMMPS atom types (1, 2).

### Scenario 4.3: Simulation Execution (End-to-End Mock)
**Priority**: Medium
**Description**: Verify the full execution flow.
**Steps**:
1.  Run `explorer.explore(potential_path="test.yace")`.
2.  (Mock) LAMMPS runs and produces output.
3.  **Check**: `work_dir/md_run/log.lammps` exists.
4.  **Check**: `work_dir/md_run/dump.lammps` exists.
5.  **Check**: The method returns a list of structures sampled from the trajectory.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: MD Exploration

  Scenario: Generating Safe Input Scripts
    Given I have a potential "mlp.yace" and need ZBL safety
    When I generate the LAMMPS input
    Then the pair style should be "hybrid/overlay"

  Scenario: Parsing Simulation Results
    Given a completed MD simulation with 5 output frames
    When I parse the dump file
    Then I should receive 5 atomic structures
    And each structure should have the correct cell dimensions
```
