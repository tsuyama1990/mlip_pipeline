# Cycle 02 UAT: The Oracle (DFT Automation)

## 1. Test Scenarios

### Scenario 01: The "Mock" Calculation
**Priority**: High
**Description**: Run the Oracle in Mock mode (no QE installed).
**Objective**: Verify the data flow from Structure -> Oracle -> Forces without needing heavy tools.

**Steps**:
1.  Configure `config.yaml` with `dft: { mode: "mock", potential: "lj" }`.
2.  Create a python script `test_mock_oracle.py`:
    ```python
    from ase.build import bulk
    from mlip_autopipec.physics.oracle.manager import DFTManager

    atoms = bulk("Cu", "fcc", a=3.6)
    oracle = DFTManager(config)
    labelled_atoms = oracle.compute([atoms])[0]
    print(f"Energy: {labelled_atoms.get_potential_energy()}")
    print(f"Forces: {labelled_atoms.get_forces()}")
    ```
3.  Run the script.
4.  **Expected Result**: Output shows a valid energy value (calculated via LJ) and zero forces (equilibrium).

### Scenario 02: Self-Healing Logic
**Priority**: Medium
**Description**: Simulate a convergence failure and verify the manager retries.
**Objective**: Test robustness.

**Steps**:
1.  In `test_healing.py`, mock the `run_command` method of the DFT runner to:
    -   Raise `DFTConvergenceError` on the 1st call.
    -   Return Success on the 2nd call.
2.  Run the test.
3.  **Expected Result**:
    -   The system logs: `[WARN] DFT convergence failed. Retrying with reduced mixing beta.`
    -   The final result is returned successfully.
    -   The Exception is consumed, not crashed.

### Scenario 03: Periodic Embedding
**Priority**: Medium
**Description**: Ensure clusters are safely boxed.
**Objective**: Avoid "lost atoms" or boundary errors.

**Steps**:
1.  Create a cluster of 2 atoms with coordinates `(0,0,0)` and `(100,0,0)` (far apart, unphysical if not handled).
2.  Pass to `DFTManager`.
3.  **Expected Result**: The Embedding logic detects the large spread or creates a box large enough (e.g., 110A side) to hold them (or raises a warning about unreasonable geometry). *Note: For standard embedding, we usually center a compact cluster.*

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: DFT Calculation Management

  Scenario: Recovering from SCF failure
    Given a structure that is hard to converge
    And the DFT settings allow for self-healing
    When the Oracle runs the calculation
    And the first attempt fails with "convergence not achieved"
    Then the Oracle should adjust "mixing_beta"
    And the Oracle should retry the calculation
    And the Oracle should eventually return valid forces

  Scenario: Periodic Embedding
    Given a non-periodic cluster of atoms
    When the Oracle prepares the input for Quantum Espresso
    Then the structure should be placed in a periodic supercell
    And the vacuum padding should be at least 10 Angstroms
```
