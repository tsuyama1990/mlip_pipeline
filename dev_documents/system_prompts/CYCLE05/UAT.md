# Cycle 05 UAT: Delta Learning and Full Pipeline Orchestration

## 1. Test Scenarios

### Scenario 01: Delta Learning Success
-   **Priority**: Critical
-   **Description**: Verify that the system can fine-tune the base potential with DFT data, effectively correcting the "Sim-to-Real gap".
-   **Execution**:
    1.  Provide a base `potential.yace` and a `dft_dataset.pckl`.
    2.  Run Step 7 (Delta Learning) with `step7_pacemaker_finetune.weight_dft: 10.0`.
    3.  Check output for "Delta Learning complete".
    4.  Verify `final_potential.yace` is created.
    5.  Load the potential and ensure it gives valid predictions (Mocked: just verify loading).

### Scenario 02: Pipeline Idempotency (Crash Recovery)
-   **Priority**: High
-   **Description**: Verify that the system can resume from a failed or interrupted state without re-running completed steps.
-   **Execution**:
    1.  Start a full pipeline run.
    2.  Manually stop the process (e.g., Ctrl+C) during Step 3.
    3.  Verify `pipeline_state.json` shows `current_step: 3`.
    4.  Restart the pipeline.
    5.  Check output for "Skipping Step 1... Skipping Step 2... Resuming Step 3".
    6.  Verify that the final output `final_potential.yace` is eventually produced.

## 2. Behavior Definitions (Gherkin)

### Feature: Full Workflow Execution

**Scenario: Execute full 7-step pipeline**
  GIVEN a valid `config.yaml`
  AND an empty workspace
  WHEN the Orchestrator executes `run_all()`
  THEN it should complete Steps 1 through 7 sequentially
  AND it should produce intermediate artifacts (`dft_dataset`, `mace.model`, `potential.yace`)
  AND it should produce the final artifact `final_potential.yace`
  AND the final potential should pass basic validation checks (not empty)

**Scenario: Resume from interruption**
  GIVEN a workspace with a `pipeline_state.json` indicating Step 3 is incomplete
  WHEN the Orchestrator executes `run_all()`
  THEN it should load the state
  AND it should verify the existence of artifacts from Steps 1 and 2
  AND it should skip Steps 1 and 2
  AND it should start execution from Step 3
  AND it should complete the remaining steps
