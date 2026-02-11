# Cycle 03 User Acceptance Test (UAT)

## 1. Test Scenarios

### Scenario 1: Mock Oracle Execution
**Priority**: P0 (Critical)
**Description**: Verify that the Mock Oracle generates `DFTResult` objects without crashing.
**Steps**:
1.  Configure `config.yaml` with `oracle.type: mock`.
2.  Provide a small dataset of structures (e.g., from Cycle 02).
3.  Run `mlip-runner compute config.yaml`.
4.  Inspect the output. Verify that `energy`, `forces` (N, 3), and `stress` (3, 3) are populated.

### Scenario 2: Periodic Embedding
**Priority**: P1 (High)
**Description**: Verify that large structures are correctly cut into smaller periodic cells.
**Steps**:
1.  Create a large supercell (e.g., 200 atoms).
2.  Manually invoke the `EmbeddingOracle.embed(atoms, center_index)` method.
3.  Inspect the resulting `embedded_atoms`.
4.  Verify that it contains only the neighbors within `R_cut + R_buffer` (approx 20 atoms).
5.  Verify that the cell is orthorhombic.

### Scenario 3: Self-Healing Mechanism (Mocked)
**Priority**: P2 (Medium)
**Description**: Verify that the Oracle retries failed calculations with adjusted parameters.
**Steps**:
1.  Configure `config.yaml` with `oracle.type: mock_fail_once` (a special mock mode).
2.  Run the computation.
3.  Check the logs.
    -   Expected: "Calculation failed for structure 001. Retrying with mixing_beta=0.3..."
    -   Expected: "Calculation succeeded on attempt 2."

## 2. Behavior Definitions (Gherkin)

### Feature: Oracle Labeling

**Scenario**: Successful Labeling (Mock)
    **Given** a list of 10 candidate structures
    **And** a configured Mock Oracle
    **When** the Orchestrator requests labeling
    **Then** 10 `DFTResult` objects should be returned
    **And** each result should contain `energy`, `forces`, and `stress`
    **And** no external DFT code should be executed

**Scenario**: Periodic Embedding of Large Structure
    **Given** a structure with 1000 atoms
    **And** an embedding radius of 6.0 Angstrom
    **When** the Embedding Oracle processes the structure centered at atom 0
    **Then** the resulting calculation cell should contain approximately 50-100 atoms
    **And** the calculation time should be significantly less than a full 1000-atom DFT run

**Scenario**: Self-Healing Retry
    **Given** a DFT calculation that fails with "SCF convergence not achieved"
    **When** the Healer intercepts the error
    **Then** the calculation should be restarted with `mixing_beta` reduced by 50%
    **And** a warning should be logged
