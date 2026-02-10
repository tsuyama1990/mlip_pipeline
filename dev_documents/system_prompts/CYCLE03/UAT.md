# Cycle 03: Oracle & DFT Automation - UAT

## 1. Test Scenarios

### Scenario 1: Basic DFT Calculation (H2 Molecule)
*   **ID**: UAT-03-001
*   **Objective**: Ensure the `DFTOracle` can run a simple calculation and return results.
*   **Pre-conditions**: `pw.x` is installed or a mock binary exists.
*   **Steps**:
    1.  Configure `OracleConfig` with `calculator_type="espresso"`.
    2.  Create an `H2` molecule with bond length 0.74 A.
    3.  Call `oracle.compute([atoms])`.
*   **Expected Result**:
    *   Returns a list with 1 structure.
    *   `atoms.info["energy"]` is approximately -30 eV (depending on pseudo).
    *   `atoms.arrays["forces"]` is present and small (< 0.1 eV/A).

### Scenario 2: Self-Healing on Convergence Failure
*   **ID**: UAT-03-002
*   **Objective**: Ensure the oracle recovers from SCF convergence errors.
*   **Pre-conditions**: Use a "Mock Calculator" that fails by default.
*   **Steps**:
    1.  Configure the `MockEspresso` to raise `CalculationFailed("convergence not achieved")` unless `mixing_beta < 0.4`.
    2.  Call `oracle.compute([atoms])`.
*   **Expected Result**:
    *   The calculation initially fails.
    *   The log shows "Retrying calculation 1/3...".
    *   The calculation succeeds on the second attempt.
    *   The result is returned correctly.

### Scenario 3: Periodic Embedding of Local Environment
*   **ID**: UAT-03-003
*   **Objective**: Verify the extraction of a cluster for DFT.
*   **Pre-conditions**: A large 1000-atom MD snapshot exists.
*   **Steps**:
    1.  Select an atom in the bulk (index 500).
    2.  Call `create_embedded_cluster(atoms, center=500, radius=6.0)`.
    3.  Check the returned structure.
*   **Expected Result**:
    *   The new structure has ~50-100 atoms.
    *   The cell is periodic and sufficiently large (e.g., > 12 A).
    *   The central atom is far from the new boundaries (> 6 A).

## 2. Behavior Definitions

```gherkin
Feature: Automated DFT Calculations

  Scenario: Successful DFT calculation
    Given I have a valid atomic structure
    And the Oracle is configured with valid pseudopotentials
    When I submit the structure to the Oracle
    Then the Oracle should run Quantum Espresso
    And the resulting structure should contain "energy", "forces", and "stress"
    And the temporary files should be cleaned up

  Scenario: Recovering from SCF divergence
    Given I have a difficult metallic system (e.g., Fe slab)
    And the default mixing beta is 0.7
    When the SCF calculation fails to converge within 100 steps
    Then the Oracle should catch the error
    And it should re-submit the job with mixing_beta = 0.3
    And if successful, it should return the valid results

  Scenario: Handling fatal errors
    Given I have a structure with overlapping atoms (distance < 0.5 A)
    When the DFT code crashes with a "segmentation fault" or "charge density negative"
    Then the Oracle should mark the structure as "FAILED"
    And it should not retry indefinitely
    And it should return a failure object or raise a specific exception
```
