# Cycle 06 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 6.1: Candidate Generation
*   **Priority**: High
*   **Goal**: Verify system can propose new data from failures.
*   **Action**:
    1.  Provide a single `Atoms` object (simulated Halt).
    2.  Invoke `candidate_generator.generate_local_candidates(atoms, n=5)`.
*   **Expectation**:
    *   Returns 5 distinct `Atoms` objects.
    *   Positions are perturbed within $\pm 0.1 \AA$.
    *   Volume is perturbed within $\pm 2\%$.

### Scenario 6.2: OTF Loop Execution (Mock)
*   **Priority**: Critical
*   **Goal**: Verify the "Brain" works.
*   **Action**:
    1.  Configure `orchestrator.max_cycles: 2`.
    2.  Configure `dynamics.mock_halt: True`.
    3.  Run `Orchestrator` cycle 4 (OTF).
*   **Expectation**:
    *   Log file shows "Dynamics halted. Generating candidates...".
    *   Log file shows "Calling Oracle...".
    *   Log file shows "Retraining potential...".
    *   Log file shows "Restarting dynamics...".
    *   Final state is `CONVERGED` (assuming Mock Dynamics eventually succeeds).

### Scenario 6.3: Convergence Check
*   **Priority**: Medium
*   **Goal**: Verify stopping criteria.
*   **Action**:
    1.  Configure `dynamics.mock_halt: False` (succeeds immediately).
    2.  Run `Orchestrator`.
*   **Expectation**:
    *   Log file shows "Dynamics converged. No retraining needed.".
    *   Loop exits early.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: On-the-Fly Learning Loop

  Scenario: System learns from mistakes
    Given a Dynamics simulation that halts due to high uncertainty
    When the Orchestrator detects the halt
    Then it should generate local candidate structures
    And acquire ground-truth labels for them
    And retrain the potential to include the new knowledge
    And resume the simulation

  Scenario: System stops when confident
    Given a Dynamics simulation that runs without halting
    When the simulation completes
    Then the Orchestrator should mark the cycle as "CONVERGED"
    And proceed to the validation phase
```
