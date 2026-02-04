# Cycle 03 UAT: The Self-Healing Oracle

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-03-01** | High | **Calculation Recovery** | Verify that the system automatically recovers from a convergence failure by adjusting parameters. |
| **UAT-03-02** | Medium | **Cluster Embedding** | Verify that a local cluster cut from a bulk structure retains the correct local geometry and is computable. |

## 2. Behavior Definitions

### UAT-03-01: Calculation Recovery

**GIVEN** a difficult electronic structure (e.g., magnetic Iron with wrong initial spin)
**AND** a `DFTManager` configured with the "Standard" strategy
**WHEN** the calculation is launched and fails with "Convergence NOT achieved" (Mocked or Real)
**THEN** the system should NOT crash
**AND** it should log "Convergence failure detected. Retrying with Strategy Level 1 (Reduced Mixing)"
**AND** it should launch a second calculation
**AND** if the second one succeeds, it should return the results

### UAT-03-02: Cluster Embedding

**GIVEN** a 1000-atom MD snapshot of MgO with a grain boundary
**WHEN** the `EmbeddingEngine` is asked to extract a 50-atom cluster around the boundary
**THEN** the returned `Atoms` object should have approximately 50 atoms + buffer
**AND** the `cell` should be orthorhombic
**AND** the periodic boundary conditions should be set to `True` (creating a "bulk-like" supercell of the defect)
