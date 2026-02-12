# Cycle 05 UAT: Dynamics Engine (MD)

## 1. Test Scenarios

### Scenario 1: Hybrid Potential Execution
*   **ID**: S05-01
*   **Goal**: Verify that MD runs using the Hybrid (ACE + ZBL) potential.
*   **Priority**: Critical.
*   **Steps**:
    1.  Create a simple MgO structure.
    2.  Provide a valid `potential.yace`.
    3.  Run `DynamicsEngine.run(structure, potential)`.
    4.  Check `in.lammps`:
        *   `pair_style hybrid/overlay pace zbl` must be present.
        *   `pair_coeff * * zbl` must be correctly set (Mg=12, O=8).
    5.  Assert `result.halted` is False (for a stable run).

### Scenario 2: Uncertainty Watchdog (Halt)
*   **ID**: S05-02
*   **Goal**: Verify that simulation halts when uncertainty spikes.
*   **Priority**: High.
*   **Steps**:
    1.  Configure `uncertainty_threshold: 5.0`.
    2.  Use a mock potential or mock LAMMPS run that outputs `max_gamma = 10.0`.
    3.  Run `DynamicsEngine.run()`.
    4.  Assert `result.halted` is True.
    5.  Assert `result.halt_reason` is "High Uncertainty".

### Scenario 3: Halt Diagnosis
*   **ID**: S05-03
*   **Goal**: Verify that the halted snapshot is correctly extracted.
*   **Priority**: Medium.
*   **Steps**:
    1.  Given a halted run (Scenario 2).
    2.  Call `HaltDiagnoser.extract(dump_file)`.
    3.  Assert it returns a `Structure`.
    4.  Assert this structure corresponds to the last frame.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Molecular Dynamics Execution

  Scenario: Running with Hybrid Potential
    Given a valid ACE potential file
    And a target structure "MgO"
    When I run an MD simulation
    Then the input script should use "pair_style hybrid/overlay"
    And the ZBL baseline should be active
    And the simulation should complete without errors

  Scenario: Halting on High Uncertainty
    Given the uncertainty threshold is set to 5.0
    And the simulation encounters a configuration with gamma = 8.0
    When the watchdog detects this
    Then the simulation should stop immediately
    And the result should indicate "halted"
```
