# Cycle 06 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario 6.1: The Closed Loop (Mock Mode)
**Priority**: Critical
**Description**: Verify that the entire system works as a cohesive unit.
**Steps**:
1.  Configure `max_cycles = 3`.
2.  Run `mlip-pipeline run`.
3.  Monitor logs.
4.  **Expectation**:
    *   Cycle 1 starts -> Exploration -> Oracle -> Training -> Validation -> Cycle 1 ends.
    *   Cycle 2 starts ...
    *   Cycle 3 starts ...
    *   Process finishes with "Pipeline Completed".

### Scenario 6.2: Checkpoint & Resume
**Priority**: High
**Description**: Verify that the system can pick up where it left off.
**Steps**:
1.  Run the pipeline with `max_cycles=5`.
2.  Manually stop the process (Ctrl+C) during Cycle 3.
3.  Check `work_dir/state.json`. It should indicate Cycle 3 is in progress or done.
4.  Run `mlip-pipeline run --resume`.
5.  **Expectation**: The logs say "Resuming from Cycle 3..." and continue to Cycle 5.

### Scenario 6.3: Data Accumulation Verification
**Priority**: Medium
**Description**: Verify that the dataset actually grows.
**Steps**:
1.  Run the full loop.
2.  Inspect `work_dir/data/dataset_stats.json` (if available) or the logs.
3.  **Check**: "Dataset size: 100 -> 200 -> 300..."

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Active Learning Orchestration

  Scenario: Completing Multiple Cycles
    Given a configuration with max_cycles = 3
    When I run the pipeline
    Then the orchestrator should execute 3 full iterations
    And the final potential should be version 3

  Scenario: Resuming from Interruption
    Given a pipeline that was stopped at Cycle 2
    When I run the pipeline with the resume flag
    Then it should skip Cycle 1
    And start processing Cycle 2 immediately
```
