# Cycle 03 UAT: Oracle (DFT Automation)

## 1. Test Scenarios

These scenarios verify the DFT automation logic, focusing on error handling and cluster carving.

### Scenario 03-01: "Periodic Embedding (Cluster Cut)"
**Priority:** P1 (High)
**Description:** Verify that the system can carve out a small, calculable cluster from a large MD snapshot while preserving the local environment of a target atom.
**Success Criteria:**
-   **Input:** A 1000-atom supercell (e.g., bulk Fe).
-   **Operation:** Create a cluster centered on Atom 0 with radius 5.0 Å + 2.0 Å buffer.
-   **Output:** A new `Structure` with approx. 50-100 atoms.
-   **Check:** The distance matrix of the inner core (r < 5.0) in the cluster must match the original supercell exactly. The boundary conditions (pbc) must be True.

### Scenario 03-02: "Self-Correction Logic (Mock)"
**Priority:** P2 (Medium)
**Description:** Verify that the Oracle retries a failed calculation with relaxed parameters.
**Success Criteria:**
-   **Config:** `oracle: espresso`, `max_retries: 3`.
-   **Mock:** Simulate an SCF convergence failure (raise `EspressoError` on first attempt).
-   **Action:** The Oracle catches the error.
-   **Result:** The Oracle logs "Retrying with mixing_beta=0.3" and calls the calculator again. The second attempt succeeds.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Oracle Automation

  Scenario: Cluster Embedding from Large Cell
    Given a large supercell structure with 1000 atoms
    When I extract a cluster centered at atom 0 with radius 5.0
    Then the resulting structure should have fewer than 100 atoms
    And the local environment of the center atom should be identical to the original

  Scenario: DFT Self-Healing
    Given a DFT calculation that fails with "Convergence Error"
    When the Oracle detects the failure
    Then it should retry the calculation with "mixing_beta" reduced
    And if it succeeds, it should return the energy
```
