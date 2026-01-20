# Cycle 06: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 6.1: Full "Zero-Human" Workflow (Mini)
-   **Priority**: Critical
-   **Description**: Run the entire pipeline on a toy system (e.g., Lennard-Jones Argon or Aluminum) to verify automation.
-   **Steps**:
    1.  Config: `target: Al`, `goal: melt_quench`.
    2.  Run `mlip-auto run config.yaml`.
    3.  Let it run for 2 Generations.
-   **Success Criteria**:
    -   System generates initial structures.
    -   "DFT" runs (real or mock).
    -   Potential is trained (Gen 0).
    -   MD runs, finds uncertainty.
    -   New structures added.
    -   Potential retrained (Gen 1).
    -   Process requires NO manual commands after start.

### Scenario 6.2: Auto-Recovery from DFT Failure
-   **Priority**: High
-   **Description**: Verify the system doesn't crash when DFT fails.
-   **Steps**:
    1.  Inject a "poison pill" structure that causes QE to fail convergence (or mock the failure).
    2.  Watch the logs.
-   **Success Criteria**:
    -   System detects failure.
    -   System retries with "robust" parameters (mixing beta reduced).
    -   If it still fails, it marks the structure as "Failed" and moves on (does not crash the loop).

### Scenario 6.3: Dashboard Visualization
-   **Priority**: Low
-   **Description**: Verify the user can see progress.
-   **Steps**:
    1.  Run the pipeline for a while.
    2.  Open `dashboard/index.html`.
-   **Success Criteria**:
    -   Plots are visible.
    -   RMSE plot shows a downward trend (or at least data points).
    -   Number of structures plot is increasing.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Orchestration and Resilience

  Scenario: State Machine Transitions
    Given the system is in "Idle" state
    When new training data arrives in the database
    Then the system should transition to "Training" state
    And trigger the Pacemaker wrapper

  Scenario: Error Recovery
    Given a DFT job that fails with "convergence not achieved"
    When the Recovery Handler analyses the log
    Then it should resubmit the job with "mixing_beta" reduced
    And increment the retry counter

  Scenario: Dashboard Reporting
    Given a running active learning campaign
    When the dashboard is refreshed
    Then it should display the current Generation number
    And it should display the latest Test RMSE
```
