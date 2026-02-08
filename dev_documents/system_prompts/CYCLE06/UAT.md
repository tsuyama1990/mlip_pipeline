# Cycle 06 UAT: Validator & Full Orchestration

## 1. Test Scenarios

These scenarios verify the final end-to-end functionality, including scientific quality control.

### Scenario 06-01: "Physical Stability Check"
**Priority:** P1 (High)
**Description:** Verify that the Validator component correctly identifies physically unstable potentials.
**Success Criteria:**
-   **Config:** `validator: physics`, `elastic_tolerance: 10%`.
-   **Mock:** A "bad" potential that yields negative bulk modulus ($C_{11} < 0$).
-   **Action:** Run validation.
-   **Result:** The validation must FAIL.
-   **Output:** The report must highlight "Elastic Stability Failed".

### Scenario 06-02: "End-to-End Orchestration (The Fe/Pt Demo)"
**Priority:** P0 (Critical)
**Description:** Verify that the system can execute the full active learning loop for the Fe-Pt/MgO scenario (mocked binaries).
**Success Criteria:**
-   **Input:** Full `config.yaml` with Generator, Oracle, Trainer, Dynamics, and Validator enabled.
-   **Operation:** Run `mlip-pipeline run config.yaml`.
-   **Flow:**
    1.  **Generation:** Creates initial Fe/Pt structures.
    2.  **Oracle:** Calculates energies (mock).
    3.  **Training:** Creates potential v1.
    4.  **Dynamics:** Runs MD with v1. Simulates a halt (high uncertainty).
    5.  **Relearning:** New structures sent to Oracle -> Trainer -> potential v2.
    6.  **Validation:** Checks v2 stability.
    7.  **Convergence:** System exits successfully.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Full System Orchestration

  Scenario: Physical Stability Validation
    Given a trained potential
    When I run the physics validator
    Then it should calculate the elastic constants
    And if the Born stability criteria are violated, it should fail

  Scenario: Active Learning Loop Closure
    Given a running pipeline
    When the dynamics engine reports high uncertainty
    Then the orchestrator should trigger a new learning cycle
    And the new potential should be validated before deployment
```
