# Cycle 03 UAT: Oracle (DFT Automation)

## 1. Test Scenarios

### Scenario 1: DFT Calculation Success
*   **ID**: S03-01
*   **Goal**: Verify that the Oracle runs a standard DFT calculation and returns results.
*   **Priority**: Critical.
*   **Steps**:
    1.  Create a single water molecule structure.
    2.  Run `Oracle.compute([H2O])`.
    3.  Assert `result.success` is True.
    4.  Assert `result.energy` is a negative float.
    5.  Assert `result.forces` has shape (3, 3).

### Scenario 2: Self-Healing Mechanism (Mocked)
*   **ID**: S03-02
*   **Goal**: Verify that the Oracle retries a failed calculation with modified parameters.
*   **Priority**: High.
*   **Steps**:
    1.  Mock `BaseOracle.compute` to raise `SCFError` on first call.
    2.  Run `OracleManager.compute([structure])`.
    3.  Assert that `BaseOracle.compute` was called at least twice.
    4.  Assert that the second call had `mixing_beta=0.3`.

### Scenario 3: Periodic Embedding
*   **ID**: S03-03
*   **Goal**: Verify that large structures can be cut into small periodic supercells.
*   **Priority**: Medium.
*   **Steps**:
    1.  Create a large 100-atom supercell with a central defect.
    2.  Run `ClusterEmbedder.embed(structure, target_atoms=[50])`.
    3.  Assert the resulting structure has fewer atoms (e.g., 20).
    4.  Assert the periodic boundary conditions (pbc) are True.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: DFT Execution

  Scenario: Standard DFT Calculation
    Given a valid structure and Quantum Espresso installed
    When I request a DFT calculation
    Then I should receive an Energy value
    And I should receive Forces for all atoms
    And Stress tensor should be computed

  Scenario: Recovering from Convergence Failure
    Given the first DFT calculation fails with "convergence not achieved"
    When the Oracle manager handles the error
    Then it should retry the calculation with "mixing_beta" reduced
    And if the second attempt succeeds, return the result
```
