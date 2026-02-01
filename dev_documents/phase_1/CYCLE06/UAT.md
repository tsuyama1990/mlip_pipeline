# User Acceptance Testing (UAT): Cycle 06

## 1. Test Scenarios

Cycle 06 is about autonomy. The user starts the process and goes to sleep.

### Scenario 6.1: The Autonomous Loop
-   **ID**: UAT-C06-01
-   **Priority**: Critical
-   **Description**: Start the system from scratch. It should perform Exploration, trigger a Halt, calculate, train, and loop.
-   **Success Criteria**:
    -   Command `mlip-auto run-loop` starts.
    -   Logs show "Gen 0: Exploration".
    -   Logs show "Halt detected! Selecting candidates...".
    -   Logs show "Training new potential...".
    -   Logs show "Gen 0 -> Gen 1".

### Scenario 6.2: Resume from Interruption
-   **ID**: UAT-C06-02
-   **Priority**: High
-   **Description**: Kill the process (Ctrl+C) during the DFT phase. Restart it. It should not re-run MD. It should pick up the pending DFT jobs.
-   **Success Criteria**:
    -   Interrupt the process.
    -   Restart.
    -   Logs show "Resuming from phase CALCULATION".
    -   It does NOT show "Running Exploration".

### Scenario 6.3: Max Generations Limit
-   **ID**: UAT-C06-03
-   **Priority**: Medium
-   **Description**: Set `max_generations: 2`. Run.
-   **Success Criteria**:
    -   System runs Gen 0, Gen 1.
    -   System stops gracefully after Gen 1 Validation.
    -   Status is "COMPLETED".

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Autonomous Active Learning Loop

  Background:
    Given a valid configuration
    And a mocked environment (for speed)

  Scenario: Complete one full learning cycle
    Given the current generation is 0
    And the phase is "EXPLORATION"
    When I start the loop
    Then the LammpsRunner should be called
    And if a halt occurs, QERunner should be called
    And then PacemakerRunner should be called
    And the generation should increment to 1

  Scenario: Resume after crash
    Given a state file exists with phase "TRAINING"
    When I start the loop
    Then the system should skip Exploration and Calculation
    And immediately start the PacemakerRunner
```
