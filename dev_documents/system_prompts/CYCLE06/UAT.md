# Cycle 06: User Acceptance Testing (UAT)

## 1. Test Scenarios

Cycle 06 is about coordination. We verify the "Manager" can juggle multiple tasks and handle interruptions.

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-06-01** | High | Workflow Resumption | Verify that the system can resume from a saved checkpoint after a simulated crash. This is critical for long-running campaigns. |
| **UAT-06-02** | Medium | Parallel Execution | Verify that multiple dummy tasks are executed in parallel via Dask. |

### Recommended Notebooks
*   `notebooks/UAT_06_Workflow.ipynb`:
    1.  Initialize `WorkflowManager`.
    2.  Start a "mock" pipeline run.
    3.  Interrupt the kernel (simulate crash).
    4.  Re-initialize `WorkflowManager`.
    5.  Call `resume()`.
    6.  Verify it skips already completed steps and finishes the rest.

## 2. Behavior Definitions

### UAT-06-01: Crash Recovery

**Narrative**:
The system is in the middle of the "Seeding" phase. It has submitted 100 DFT jobs. 50 have finished and are in the DB. The power goes out. When the user restarts the system, it reads `state.json`. It sees `phase=SEEDING`. It queries the DB and finds 50 completed jobs. It checks the `pending_jobs` list. It resubmits the remaining 50 jobs (or reconnects to them if the scheduler persisted). It does *not* re-run the 50 completed ones.

```gherkin
Feature: Workflow Resilience

  Scenario: Resuming after a Crash
    GIVEN a workflow that has completed the "Generation" phase but not the "DFT" phase
    AND 50 out of 100 DFT jobs are marked as done in the DB
    WHEN the system is restarted and "resume" is called
    THEN it should load the state from "state.json"
    AND immediately proceed to checking the remaining 50 jobs
    AND it should NOT re-run the Generation phase
    AND the iteration counter should be preserved
    AND the system should eventually transition to the Training phase once all 100 are done

```

### UAT-06-02: Parallelism

**Narrative**:
We need to process 1000 structures. Serial execution would take 1000 minutes. We have a 100-core cluster. We submit 1000 jobs. We expect the wall time to be roughly 10 minutes. The user verifies that the Dask dashboard shows 100 active workers.

```gherkin
Feature: Distributed Execution

  Scenario: Running Parallel Jobs
    GIVEN a Scheduler configured with 4 workers
    WHEN 4 sleep-tasks (1 second each) are submitted
    THEN the total execution time should be approximately 1 second (not 4 seconds)
    AND all tasks should return successfully
    AND the CPU usage should indicate parallel utilization
```
