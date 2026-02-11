# Cycle 03 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 3.1: Mock Oracle Execution
*   **Priority**: High
*   **Goal**: Verify the system can label data without external dependencies.
*   **Action**:
    1.  Configure `dft.type: MOCK`.
    2.  Run `Orchestrator` cycle 2 (Labeling phase).
*   **Expectation**:
    *   `work_dir/iter_001/dft_calc/` contains labeled structures.
    *   `ase.io.read(..., index=:)` confirms `forces` array is present (non-zero).
    *   Log file says "Mock DFT completed".

### Scenario 3.2: Self-Healing Simulation
*   **Priority**: Medium
*   **Goal**: Verify error recovery logic.
*   **Action**:
    1.  Configure `dft.max_retries: 3`.
    2.  Inject a "FAIL_ONCE" flag into the Mock Oracle (via config or environment).
*   **Expectation**:
    *   Log file shows "Calculation failed. Attempt 1/3".
    *   Log file shows "Retrying with reduced mixing beta...".
    *   Final result is success.

### Scenario 3.3: Periodic Embedding Check
*   **Priority**: Low
*   **Goal**: Verify cluster preparation.
*   **Action**:
    1.  Pass a non-periodic cluster (e.g., `pbc=False`) to the Oracle.
    2.  Check the output structure.
*   **Expectation**:
    *   Output structure has `pbc=[True, True, True]`.
    *   Cell size is larger than cluster size + vacuum padding.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Oracle Computation

  Scenario: Oracle computes forces for a batch of structures
    Given a list of 5 candidate structures
    When the Oracle is invoked
    Then it should return 5 labeled structures
    And each structure should have an "energy" property
    And each structure should have a "forces" array of shape (N, 3)

  Scenario: Oracle heals a failed calculation
    Given a calculation that fails with "convergence error"
    When the Self-Healer is triggered
    Then it should suggest a new parameter set with "mixing_beta < 0.7"
    And the calculation should be retried
```
