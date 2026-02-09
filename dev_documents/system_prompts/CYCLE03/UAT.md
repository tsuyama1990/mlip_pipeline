# Cycle 03 UAT: Oracle (DFT)

## 1. Test Scenarios

### Scenario 3.1: Standard DFT Calculation (Mock)
*   **Goal:** Verify that a standard DFT calculation produces valid output.
*   **Steps:**
    1.  Create a `config.yaml` with `oracle: { type: "MOCK" }`.
    2.  Provide an input `Structure` (e.g., 2-atom Si).
    3.  Run the Oracle component (`compute()`).
*   **Expected Behavior:**
    *   The structure object is returned with populated `energy` and `forces` arrays.
    *   The `features` dict contains metadata like `dft_calculation_time`.

### Scenario 3.2: Self-Healing Logic (Failure Recovery)
*   **Goal:** Verify that the system can recover from a convergence error by adjusting parameters.
*   **Steps:**
    1.  Create a `config.yaml` with `oracle: { type: "MOCK_FAIL_ONCE" }`.
    2.  Provide an input structure.
    3.  Run the Oracle component.
    4.  Inspect logs for `WARNING: DFT convergence failed. Retrying...`.
*   **Expected Behavior:**
    *   The calculation completes successfully on the second attempt.
    *   The second attempt uses safer parameters (e.g., reduced mixing beta).

### Scenario 3.3: Periodic Embedding
*   **Goal:** Verify that a local cluster is correctly extracted from a large system.
*   **Steps:**
    1.  Create a large `ase.Atoms` object (e.g., 500 atoms, random positions).
    2.  Select a central atom (index 0).
    3.  Run the `PeriodicEmbedding` logic with `cutoff=5.0 Å` and `buffer=2.0 Å`.
    4.  Inspect the output cluster.
*   **Expected Behavior:**
    *   The output cluster has significantly fewer atoms (e.g., 50-100).
    *   The central atom and all neighbors within 5.0 Å are present.
    *   The cell dimensions are appropriate (e.g., cubic box large enough to prevent self-interaction, or true periodic supercell).

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: DFT Oracle

  Scenario: Calculate forces on a structure
    Given a valid atomic structure "Si_bulk"
    And a DFT calculator "MockQuantumEspresso"
    When I request a calculation
    Then the structure should have "potential_energy" defined
    And the structure should have "forces" array of shape (N, 3)

  Scenario: Recover from SCF convergence failure
    Given a structure that causes convergence issues
    And a self-healing strategy "ReduceMixing"
    When the first calculation fails with "SCFError"
    Then the system should automatically retry
    And the retry should use mixing_beta = 0.3
    And the final result should be valid

  Scenario: Extract local environment (Embedding)
    Given a large MD snapshot of 1000 atoms
    And a target atom index 42
    When I extract the local cluster with radius 5.0 Å
    Then the resulting structure should contain only atoms within approximately 7.0 Å (radius + buffer)
    And the local topology around atom 42 should match the original snapshot
```
