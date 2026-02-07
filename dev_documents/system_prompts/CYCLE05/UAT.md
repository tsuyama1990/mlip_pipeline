# Cycle 05 UAT: Dynamics & On-the-Fly (OTF) Engine

## 1. Test Scenarios

### SCENARIO 01: Hybrid Potential Configuration
**Priority**: High
**Description**: Verify that the generated LAMMPS input enables the hybrid potential mode (ACE + ZBL) for stability.
**Pre-conditions**: `DynamicsEngine` initialised.
**Steps**:
1.  Call `_generate_input_script(potential="test.yace")`.
2.  Inspect the string output.
**Expected Result**:
-   Contains `pair_style hybrid/overlay pace zbl ...`.
-   Contains `pair_coeff * * pace test.yace ...`.
-   Contains `pair_coeff * * zbl ...`.

### SCENARIO 02: Uncertainty Halt Trigger
**Priority**: High
**Description**: Verify that the engine correctly handles an uncertainty-driven halt from LAMMPS.
**Pre-conditions**: A mock LAMMPS execution that exits with "Fix halt condition met".
**Steps**:
1.  Run `run_exploration()`.
2.  (Mock behaviour: write "Fix halt condition met" to stdout and exit code 1).
3.  Check the return object `ExplorationResult`.
**Expected Result**:
-   `status` is `ExplorationStatus.HALTED`.
-   `final_structure` is not None.
-   Logs indicate "Uncertainty limit reached, capturing structure...".

## 2. Behaviour Definitions

```gherkin
Feature: Dynamics Engine

  Scenario: Generating robust inputs
    Given a configuration for a hybrid simulation
    When I generate the LAMMPS script
    Then it should strictly overlay ZBL on top of ACE
    And it should define the "max_gamma" variable for monitoring

  Scenario: Handling simulation crash (non-halt)
    Given a simulation that crashes due to segmentation fault (bad physics)
    When I run the exploration
    Then the engine should catch the error
    And it should return status "CRASHED"
    And it should NOT return status "HALTED" (distinguish bugs from uncertainty)
```
