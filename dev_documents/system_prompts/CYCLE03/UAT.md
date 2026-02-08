# Cycle 03 UAT: Oracle & DFT Interface

## 1. Test Scenarios

### Scenario 3.1: Quantum Espresso K-point Generation
*   **ID**: S03-01
*   **Priority**: High
*   **Description**: Verify automatic k-point grid generation based on cell size.
*   **Steps**:
    1.  Create a script with a large supercell (e.g., 20 Å cube) and a small primitive cell (e.g., 3 Å cube).
    2.  Set `kspacing = 0.04`.
    3.  Call `k_grid_from_spacing`.
*   **Expected Result**:
    *   Large cell: `[1, 1, 1]` grid.
    *   Small cell: `[N, N, N]` where $N \approx 1/(3 \times 0.04) \approx 8$.

### Scenario 3.2: Self-Healing Mechanism (Mocked)
*   **ID**: S03-02
*   **Priority**: Critical
*   **Description**: Verify the Oracle retries the calculation with different parameters upon failure.
*   **Steps**:
    1.  Create a `MockQE` that is programmed to fail the first attempt with "convergence not achieved" and succeed the second time.
    2.  Call `oracle.compute([structure])`.
*   **Expected Result**:
    *   The Oracle should catch the first error.
    *   The Oracle should re-submit with lower `mixing_beta`.
    *   The final result should be a labeled structure.
    *   Logs should show "Retrying calculation with mixing_beta=0.3".

### Scenario 3.3: Periodic Embedding
*   **ID**: S03-03
*   **Priority**: Medium
*   **Description**: Verify a cluster is correctly placed in a padded box.
*   **Steps**:
    1.  Take a 13-atom cluster (radius ~3 Å).
    2.  Call `embedding.embed_cluster(cluster, padding=5.0)`.
*   **Expected Result**:
    *   The resulting cell size should be approximately $2 \times (3+5) = 16$ Å.
    *   Atoms should be centered.
    *   PBC should be True.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: DFT Oracle

  Scenario: Automatic K-point selection
    GIVEN a silicon primitive cell (a=5.43 Angstrom)
    AND a target k-spacing of 0.04 inverse Angstrom
    WHEN I request the k-point grid
    THEN the grid size should be at least [4, 4, 4]

  Scenario: Recovering from SCF failure
    GIVEN a difficult electronic structure (e.g., magnetic Iron)
    WHEN the DFT calculation fails with "convergence error"
    THEN the Oracle should automatically restart with "mixing_beta" reduced
    AND eventually return a valid energy
```
