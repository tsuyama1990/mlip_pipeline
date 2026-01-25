# Cycle 02 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 2.1: Single Structure Calculation
*   **Priority:** High
*   **Description:** The system takes a single atomic structure (e.g., Silicon bulk), runs a DFT calculation, and correctly retrieves the Potential Energy and Forces.
*   **Pre-requisites:** Quantum Espresso (`pw.x`) installed or a mock script provided.
*   **Input:** `POSCAR` or ASE Atoms object for Si.
*   **Expected Output:**
    *   A `DFTResult` object.
    *   `energy` is a float value (approx -308 eV for 8 atom Si).
    *   `forces` is an array of near-zeros (for equilibrium).
    *   `success` is True.

### Scenario 2.2: Recovery from Divergence
*   **Priority:** High
*   **Description:** The system encounters a structure that causes SCF divergence with default settings. It should automatically detect this and retry with more robust settings (e.g., smaller mixing beta) to achieve convergence.
*   **Input:** A "difficult" structure (e.g., atoms very close together).
*   **Simulation:** Can be simulated by a mock that fails unless `mixing_beta < 0.3`.
*   **Expected Output:**
    *   Logs showing "SCF failed, retrying with mixing_beta=0.3".
    *   Final result is `success=True`.

### Scenario 2.3: Graceful Failure
*   **Priority:** Medium
*   **Description:** If a structure is physically impossible (e.g., atoms fully overlapping) and fails all recovery attempts, the system should return a `DFTResult` with `success=False` rather than crashing the Python process.
*   **Input:** Two atoms at distance 0.1 Å.
*   **Expected Output:**
    *   `DFTResult` with `success=False`.
    *   Error message captured in the result.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: DFT Execution

  Scenario: Successful Calculation
    Given a valid Silicon structure
    And the DFT command is configured correctly
    When I request a DFT calculation
    Then the result should contain a valid Energy
    And the result should contain Forces of shape (N, 3)

  Scenario: Auto-Recovery
    Given a structure that fails SCF with default mixing
    When I request a DFT calculation
    Then the runner should detect the failure
    And the runner should retry with reduced mixing beta
    And the final result should be successful
```
