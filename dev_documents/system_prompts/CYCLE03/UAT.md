# CYCLE 03 UAT: Oracle & Self-Healing

## 1. Test Scenarios

### Scenario 03.1: Self-Healing Mechanism
*   **Priority**: High
*   **Objective**: Verify that the Oracle recovers from SCF convergence failures.
*   **Description**: Simulate a convergence failure in DFT and verify that the system automatically retries with more robust parameters.
*   **Input**:
    *   Mock Calculator configured to fail twice then succeed.
*   **Success Criteria**:
    *   Logs show "SCF Failed. Retrying with mixing_beta=0.3".
    *   Logs show "SCF Failed. Retrying with smearing=0.02".
    *   Final result is returned successfully.

### Scenario 03.2: Periodic Embedding
*   **Priority**: Medium
*   **Objective**: Verify that non-periodic clusters are correctly handled.
*   **Description**: Feed an isolated atom (with no cell defined) to the Oracle.
*   **Success Criteria**:
    *   The structure passed to the DFT calculator has a defined unit cell (e.g., 20x20x20 Å box).
    *   The atom is centered in the box.

## 2. Behavior Definitions

```gherkin
Feature: DFT Oracle

  Scenario: Automatic Error Recovery
    GIVEN a DFT calculation that fails to converge with default settings
    WHEN the Oracle detects the "Convergence NOT achieved" error
    THEN it should reduce the mixing beta parameter
    AND restart the calculation without user intervention

  Scenario: Embedding Isolated Clusters
    GIVEN a structure generated with no periodic boundary conditions (pbc=False)
    WHEN the Oracle prepares the input
    THEN it should embed the structure in a vacuum box
    AND set pbc=True
    AND ensure the vacuum size is at least twice the potential cutoff
```
