# Cycle 06 UAT: Orchestrator & Active Learning Loop

## 1. Test Scenarios

### Scenario 1: Full Active Learning Cycle (Mocked)
*   **ID**: S06-01
*   **Goal**: Verify that the Orchestrator runs a full cycle of Explore -> Label -> Train.
*   **Priority**: Critical.
*   **Steps**:
    1.  Mock `Generator`, `Oracle`, `Trainer`, `Dynamics`.
    2.  Mock `Dynamics.run()` to return `halted=True` (simulate discovery).
    3.  Mock `Trainer.train()` to return a new `.yace` file.
    4.  Run `Orchestrator.run()`.
    5.  Assert `oracle.compute()` was called.
    6.  Assert `trainer.train()` was called.
    7.  Assert `state.iteration` incremented.

### Scenario 2: Convergence Check
*   **ID**: S06-02
*   **Goal**: Verify that the loop terminates when no high uncertainty is found.
*   **Priority**: High.
*   **Steps**:
    1.  Mock `Dynamics.run()` to return `halted=False`.
    2.  Run `Orchestrator.run()`.
    3.  Assert `oracle.compute()` was NOT called.
    4.  Assert loop exits gracefully.

### Scenario 3: Candidate Selection Integration
*   **ID**: S06-03
*   **Goal**: Verify that halted structures are perturbed and filtered.
*   **Priority**: Medium.
*   **Steps**:
    1.  Given a halted structure with 1 high-gamma atom.
    2.  Mock `CandidateSelector.select()`.
    3.  Run the loop.
    4.  Assert `oracle.compute(candidates)` receives a list of perturbed structures.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Orchestrator Loop

  Scenario: Handling Discovery (Halt Event)
    Given the current potential is uncertain (mocked halt)
    When the orchestrator runs a cycle
    Then it should select new candidate structures
    And label them using the Oracle
    And update the potential using the Trainer
    And increment the iteration count

  Scenario: Handling Convergence
    Given the current potential is robust (no halt)
    When the orchestrator runs a cycle
    Then it should log "Converged"
    And terminate the loop without further training
```
