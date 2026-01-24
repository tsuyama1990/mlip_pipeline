# Cycle 08 User Acceptance Testing (UAT) Plan

## 1. Test Scenarios

### Scenario ID: UAT-C08-001 - The "Set It and Forget It" Run (Full Autonomy)

**Priority:** High
**Description:**
The ultimate goal. The user sets up a project, types one command, and walks away. The system runs a full generation -> train -> explore loop.

**User Story:**
As a PhD Student, I want to run `mlip-auto loop` on Friday evening, so that when I come back on Monday, I have a converged potential and a dataset of 5,000 structures, without me having to intervene for crashes or file management.

**Step-by-Step Walkthrough:**
1.  **Preparation**: User initializes a clean project `Al_Loop`.
2.  **Configuration**: `input.yaml` set to `max_iterations: 3`.
3.  **Execution**: `mlip-auto loop > run.log 2>&1 &` (Background run).
4.  **Monitoring (Monday Morning)**:
    -   User checks `run.log`.
    -   *Log*: "Iteration 1 Complete."
    -   *Log*: "Iteration 2 Complete."
    -   *Log*: "Iteration 3 Complete. Stopping."
5.  **Artifacts**:
    -   `mlip.db`: Contains ~1000 structures (from Gen and Active Learning).
    -   `potentials/`: Contains `potential_iter_0.yace`, `potential_iter_1.yace`, `potential_iter_2.yace`.
    -   `state.json`: Shows `current_phase: DONE`.

**Success Criteria:**
-   The loop executed 3 times.
-   Data flowed correctly between modules (Gen -> DFT -> Train -> Inf -> DFT).
-   State was persisted (if I killed it halfway and restarted, it should have resumed).

### Scenario ID: UAT-C08-002 - Graceful Interruption and Resume

**Priority:** High
**Description:**
HPC clusters often have time limits (walltime). The job might be killed. It must resume.

**User Story:**
As a User, I need the system to checkpoint its state, so that if the 24-hour walltime hits during "DFT Phase", the next submitted job picks up exactly where it left off, rather than restarting from Generation.

**Step-by-Step Walkthrough:**
1.  **Start**: Run `mlip-auto loop`.
2.  **Interrupt**: Press `Ctrl+C` (or `kill`) during "DFT Phase" (simulated by a long sleep in the mock).
3.  **Check State**: `state.json` says `current_phase: DFT`, `iteration: 1`.
4.  **Resume**: Run `mlip-auto loop`.
5.  **Observation**:
    -   CLI: "Resuming from Iteration 1, Phase DFT."
    -   CLI: "Found 40 Pending jobs. Resuming..."
    -   *NOT*: "Generating initial structures..." (It skips Generation because it was done).

**Success Criteria:**
-   The system correctly identifies the saved state.
-   It skips completed phases.
-   It finishes the loop.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Workflow Orchestration
  As a Conductor
  I want to manage the active learning cycle
  So that the process is autonomous and robust

  Scenario: Full Loop Execution
    Given a valid configuration with max_iterations = 2
    When I start the loop
    Then the system should execute Generation Phase
    And then the DFT Phase
    And then the Training Phase
    And then the Inference Phase
    And then loop back to DFT Phase (for Iteration 2)
    And finally stop after Iteration 2

  Scenario: Resume Capability
    Given a workflow that was interrupted during Training
    And the state file records "Phase: TRAINING"
    When I restart the loop
    Then the system should immediately start the Training Phase
    And it should NOT re-run Generation or DFT
```
