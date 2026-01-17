# Cycle 08 UAT: Orchestration & Production Readiness

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-08-01** | High | **End-to-End Autonomous Run** | Verify that the system can perform a complete "Zero-Human" run: starting from an input file, iterating through generation, DFT, training, and inference, and terminating successfully. |
| **UAT-08-02** | High | **Checkpoint & Resume** | Verify that if the execution is interrupted (e.g., Ctrl+C or power failure), restarting the command resumes from the last stable state (e.g., skipping completed DFT calculations) rather than starting over. |
| **UAT-08-03** | Medium | **Parallel Execution** | Verify that DFT tasks are submitted in parallel (e.g., using Dask) and not sequentially. |
| **UAT-08-04** | Low | **Dashboard Monitoring** | Verify that the system produces a `report.html` (or similar) that updates in real-time, allowing a user to monitor the learning curve and job status. |

### Recommended Demo
Create `demo_08_full_pipeline.ipynb`.
1.  **Block 1**: Setup a `SystemConfig` configured to use "Mock Engines" (fast execution).
2.  **Block 2**: Initialize `WorkflowManager`.
3.  **Block 3**: Run `manager.run()`. Capture the live logs in the notebook.
4.  **Block 4**: Interrupt the kernel manually during "Generation 1".
5.  **Block 5**: Re-run `manager.run()`. Verify the logs say "Resuming from Generation 1...".
6.  **Block 6**: Allow it to finish. Display the final `report.html` (or `learning_curve.png`) inline.

## 2. Behavior Definitions

### Scenario: The Full Loop
**GIVEN** a clean project directory and a valid input config.
**WHEN** the manager runs.
**THEN** it should produce `potential_gen0.yace`.
**AND** it should run inference.
**AND** if inference finds errors, it should produce `potential_gen1.yace`.
**AND** the final generation should have lower uncertainty than the first.
**AND** the database should contain records from both generations.
**AND** the log should show "Pipeline Finished Successfully".

### Scenario: Resilience
**GIVEN** a batch of 100 DFT jobs submitted to Dask.
**WHEN** 5 of them fail (mocked failure).
**THEN** the workflow should NOT crash.
**AND** the training should proceed with the 95 successful ones.
**AND** the logs should report "5 jobs failed".

### Scenario: Visualization
**GIVEN** a run that has completed 3 generations.
**WHEN** the dashboard is updated.
**THEN** the RMSE plot should show 3 data points.
**AND** the "Total Structures" counter should equal the sum of Gen0 + Gen1 + Gen2 data.

### Scenario: Resource Management
**GIVEN** a config specifying `parallel_cores=4`.
**WHEN** the Dask cluster starts.
**THEN** it should show 4 workers (or threads).
**AND** `submit_dft_batch` should not block the main thread.
