# Cycle 02 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 2.1: The "Mock Loop" Orchestration
**Goal**: Verify that the `SimpleOrchestrator` correctly manages the Active Learning loop with mocks.
**Priority**: Critical (P0) - Core logic of the system.
**Steps**:
1.  Configure `config.yaml` with `type: mock` for all components.
2.  Set `max_cycles: 2`.
3.  Execute the CLI command `pyacemaker run --config config.yaml`.
**Success Criteria**:
*   The script runs to completion (exit code 0).
*   The console logs show:
    *   "Starting Iteration 0..."
    *   "Dynamics Halted!"
    *   "Generating Candidates..."
    *   "Oracle Computed Energy..."
    *   "Training Potential..."
    *   "Validation Passed."
    *   "Starting Iteration 1..."
*   A `dataset.json` file is created and contains structures from both iterations.
*   Two potential files (e.g., `potential_0.yace`, `potential_1.yace`) are created.

### Scenario 2.2: Data Persistence & Recovery
**Goal**: Verify that the orchestrator can save its state and resume (or at least handle restarts gracefully).
**Priority**: High (P1) - Prevents data loss.
**Steps**:
1.  Run the loop for 1 cycle.
2.  Inspect the `dataset.json` file.
3.  Manually edit the file (simulate data corruption or manual addition).
4.  Run the loop again, pointing to the same workdir.
**Success Criteria**:
*   The system loads the existing dataset without overwriting it.
*   New data is appended correctly.
*   If the file is corrupted, it should raise a clear `ValidationError` instead of crashing silently or corrupting memory.

## 2. Behaviour Definitions (Gherkin)

```gherkin
Feature: Orchestration & Data

  Scenario: Basic Active Learning Loop
    Given a clean work directory
    And a config with max_cycles=2
    When I run the orchestrator
    Then the loop should run exactly 2 times
    And the dataset sise should increase after each cycle
    And the potential version should increment

  Scenario: Resume from Checkpoint (Dataset)
    Given an existing dataset with 10 structures
    When I start the orchestrator
    Then it should load the 10 structures
    And continue training from that point
    And the final dataset should have > 10 structures
```
