# Cycle 08 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 8.1: kMC Exploration (EON)
*   **Priority:** High
*   **Description:** The system runs a kMC simulation to find saddle points and escape local minima.
*   **Input:** A metastable structure.
*   **Expected Output:**
    *   EON starts and runs process searches.
    *   If a high-uncertainty saddle point is found, the system halts and captures it.

### Scenario 8.2: Full "Zero-Config" Run
*   **Priority:** Critical
*   **Description:** The ultimate test. The user starts the system with minimal input, and it runs for 2-3 cycles autonomously.
*   **Input:** `config.yaml` and `POSCAR`.
*   **Execution:** Run for 2 cycles (mocked DFT/Training to save time).
*   **Expected Output:**
    *   Cycle 1 completes.
    *   Cycle 2 completes.
    *   `potentials/` directory contains `generation_001.yace`, `generation_002.yace`.
    *   `validation_reports/` contains reports for each generation.

### Scenario 8.3: Resume Capability
*   **Priority:** Medium
*   **Description:** If the system is interrupted in Cycle 4, restarting it should resume from Cycle 4, not Cycle 1.
*   **Input:** Existing workspace with `state.json` indicating Cycle 4.
*   **Expected Output:**
    *   Logs show "Resuming from Cycle 4".

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Full System Orchestration

  Scenario: Run kMC with Active Learning
    Given a working potential
    When I enable EON exploration
    Then the system should explore transition states
    And if a rare event triggers high uncertainty
    Then the system should capture the transition state for DFT

  Scenario: Autonomous Loop
    Given a fresh workspace
    When I start the PyAcemaker system
    Then it should perform Structure Generation
    And it should perform DFT calculations
    And it should train a potential
    And it should validate the potential
    And it should repeat this cycle until convergence
```
