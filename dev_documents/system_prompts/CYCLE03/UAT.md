# Cycle 03: Oracle (DFT) UAT

## 1. Test Scenarios

### Scenario 03-01: Basic DFT Calculation
**Priority**: High
**Goal**: Verify QE integration.
**Description**:
Calculate the energy and forces for a single Bulk Silicon structure (2 atoms).
**Expected Outcome**:
-   The calculation finishes (exit code 0).
-   The `Structure` object has `energy` (float) and `forces` (Nx3 array).
-   The forces on ideal bulk atoms should be nearly zero (< 0.01 eV/A).

### Scenario 03-02: Self-Healing Mechanism
**Priority**: Critical
**Goal**: Verify robustness against convergence failure.
**Description**:
1.  Configure QE with a very high mixing beta (e.g., 1.0) or aggressive algorithm that is likely to diverge for a complex system (e.g., Magnetic Fe).
2.  Run `Oracle.compute()`.
3.  Inject a mock failure if real QE is too stable.
**Expected Outcome**:
-   The logs show "SCF Convergence Failed -> Retrying with mixing_beta=0.3".
-   The calculation eventually succeeds.

### Scenario 03-03: Periodic Embedding Visualization
**Priority**: Medium
**Goal**: Verify cluster extraction logic.
**Description**:
1.  Take a large supercell (e.g., 4x4x4 Ag) with a vacancy.
2.  Extract a cluster of radius 6.0 A around the vacancy.
3.  Embed it into a vacuum box.
4.  Visualize the result.
**Expected Outcome**:
-   A new `Structure` object containing only ~50 atoms.
-   The vacancy is centered.
-   The box size is large enough to prevent self-interaction (> 12.0 A).

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Oracle Computation

  Scenario: Calculate forces for Silicon
    Given a bulk Silicon structure
    When I request DFT calculation with "low" precision
    Then the energy should be approximately -100 eV (mock value)
    And the forces should be computed

  Scenario: Heal a failed calculation
    Given a structure that causes SCF divergence
    When I request DFT calculation
    Then the Oracle should retry with different parameters
    And the final result should be valid
```

## 3. Jupyter Notebook Validation (`tutorials/02_Oracle_Test.ipynb`)
-   **Structure**: Load a structure from `ase.build`.
-   **Run**: `oracle.compute([structure])`.
-   **Inspect**:
    -   `print(structure.energy)`
    -   `print(structure.forces)`
    -   `view(structure)` (if using `ase.visualize`).
