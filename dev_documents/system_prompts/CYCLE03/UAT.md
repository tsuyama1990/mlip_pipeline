# Cycle 03 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario 3.1: Automated DFT Calculation (Mock)
*   **Goal**: Verify the Oracle can take a structure, run a "DFT calculation" (Mock), and return energy/forces.
*   **Action**:
    1.  User runs `pyacemaker run-loop` with `oracle.type: mock`.
    2.  User verifies the orchestrator logs show "Running DFT on 10 structures".
    3.  User verifies that `SinglePointResult` objects are created with `energy` and `forces` arrays of correct shape (Nx3).
*   **Success Criteria**:
    *   No crashes.
    *   Energy is a float.
    *   Forces match the number of atoms.

### Scenario 3.2: Self-Healing Logic (Failure Recovery)
*   **Goal**: Verify that the system automatically retries a failed calculation with adjusted parameters.
*   **Action**:
    1.  User configures `oracle.mock_failure_rate: 0.5` (a new config option for testing).
    2.  User runs the loop.
    3.  User observes the logs for "DFT failed. Retrying with mixing_beta=0.3...".
*   **Success Criteria**:
    *   The system does not crash on the first failure.
    *   The log shows at least one successful retry.
    *   The final result is marked as `converged=True`.

### Scenario 3.3: Periodic Embedding (Cluster Construction)
*   **Goal**: Verify that a local cluster is correctly extracted from a large structure and embedded in a periodic box.
*   **Action**:
    1.  User creates a script to test `embedding.create_periodic_cluster`.
    2.  Input: A large 10x10x10 supercell with one defect atom at the center.
    3.  User runs the script with $R_{cut}=5.0$.
*   **Success Criteria**:
    *   The output structure has a smaller number of atoms (only neighbors within cutoff).
    *   The cell vectors are orthogonal and large enough to prevent self-interaction ($> 2 \times R_{cut}$).
    *   PBC is set to True.

## 2. Behavior Definitions (Gherkin Style)

### Feature: DFT Calculation
**Scenario**: Successful energy/force calculation
  **Given** an atomic structure
  **When** the Oracle computes the property
  **Then** it should return a SinglePointResult
  **And** the result should contain potential energy and atomic forces

### Feature: Self-Healing
**Scenario**: Convergence failure handling
  **Given** a structure that causes SCF convergence failure
  **When** the Oracle encounters the error
  **Then** it should log the failure
  **And** it should retry the calculation with a smaller mixing beta
  **And** if successful, return the result

### Feature: Periodic Embedding
**Scenario**: Creating a cluster model
  **Given** a large supercell with a central atom of interest
  **When** the embedding function is called with a cutoff radius
  **Then** a new smaller Atoms object should be returned
  **And** it should contain only atoms within the cutoff + buffer
  **And** the cell should be periodic and sufficiently large
