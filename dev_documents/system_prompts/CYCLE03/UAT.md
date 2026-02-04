# CYCLE 03 UAT: Oracle & Data Management

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-03-01** | High | Self-Healing DFT | Verify that the Oracle recovers from simulated SCF convergence failures by adjusting parameters. |
| **UAT-03-02** | High | Periodic Embedding | Verify that a local cluster extracted from a large system is correctly wrapped into a periodic supercell for DFT. |
| **UAT-03-03** | Medium | Force Masking Preparation | Verify that the returned atoms contain information (e.g., tags or arrays) distinguishing the "core" atoms from the "buffer" atoms. |

## 2. Behavior Definitions

### Scenario: Self-Healing Recovery
**GIVEN** an `EspressoOracle` configured with a retry strategy
**AND** a mocked calculator that fails twice before succeeding
**WHEN** `calculate()` is called
**THEN** the system should log "Retrying with reduced mixing_beta"
**AND** finally return the calculated energy without crashing.

### Scenario: Periodic Embedding
**GIVEN** a 1000-atom supercell with a defect in the center
**WHEN** `embed_cluster` is called with $R_{cut}=5.0$ and $R_{buffer}=2.0$
**THEN** the resulting structure should have significantly fewer atoms (e.g., < 100)
**AND** the cell dimensions should be tight around the atoms
**AND** `pbc` should be true in all directions.
