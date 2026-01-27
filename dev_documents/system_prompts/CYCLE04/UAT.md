# Cycle 04 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario ID: UAT-04-01 (Autonomous Loop)
**Priority**: Critical
**Description**: Verify that the system can run multiple cycles without human intervention.
**Steps**:
1.  Setup a config with `max_cycles: 2`.
2.  Run `mlip-auto run-loop`.
3.  Monitor the process.
4.  Expect:
    -   Cycle 1: Exploration -> Halt -> DFT -> Train.
    -   Cycle 2: Exploration (using new potential) -> Halt -> DFT -> Train.
    -   Process exits successfully.

### Scenario ID: UAT-04-02 (Crash Recovery)
**Priority**: High
**Description**: Verify that the system can recover from a crash.
**Steps**:
1.  Start a loop.
2.  Wait until it reaches the "DFT Calculation" phase.
3.  Kill the process (`kill -9`).
4.  Restart the command.
5.  Expect:
    -   System prints "Resuming from cycle X, phase Calculation".
    -   It does *not* restart Exploration.
    -   It continues to finish the cycle.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Active Learning Orchestration

  Scenario: Full autonomous cycle
    GIVEN a fresh configuration
    WHEN I start the loop
    THEN the system should perform exploration
    AND if uncertainty is detected, it should automatically trigger DFT
    AND it should update the potential
    AND it should increment the iteration counter
    AND it should stop after max_cycles is reached
```
