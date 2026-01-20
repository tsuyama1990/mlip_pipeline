# Cycle 06: Orchestration & Final Polish

## 1. Summary

Cycle 06 marks the completion of the MLIP-AutoPipe development roadmap. In Cycles 1-5, we successfully built the individual engines required for active learning: the Structure Generator, the DFT Factory, the Surrogate Filter, the Pacemaker Trainer, and the Inference Engine. However, these components currently exist as isolated tools. A human operator is still required to move files, check logs, and run commands.

In this final cycle, we implement **Module F: Orchestration**. This module integrates the disparate components into a unified, autonomous system—the **"Zero-Human" Protocol**. The goal is to create a system where the user provides a single configuration file (e.g., "Learn Fe-Ni alloy"), and the software autonomously loops through generation, labeling, training, and exploration until a converged potential is produced.

The core components of this cycle are:
1.  **Workflow Manager**: A robust State Machine that tracks the global status of the campaign. It decides when to transition from "Exploration" to "Labeling" based on data accumulation thresholds, and when to transition to "Training" based on queue status.
2.  **Auto-Recovery**: We implement "Expert System" logic to handle failures in external tools. If Quantum Espresso fails to converge, the system catches the error, analyzes the output, modifies the mixing parameters or diagonalization algorithm, and resubmits the job automatically. This turns a fragile script into a resilient daemon.
3.  **Task Queue**: We replace simple loops with a scalable `TaskQueue` (based on `dask.distributed`), allowing the system to utilize all available cores on a workstation or scale to a cluster.
4.  **Dashboard**: We provide a web-based dashboard (using `Plotly`) that allows the user to visualize the "Brain" of the system: plotting the RMSE learning curve, the phase space coverage, and the history of calculations.

By the end of this cycle, the software will be a polished, production-ready tool that democratizes access to high-accuracy ML potentials.

## 2. System Architecture

### 2.1. Code Blueprint and File Structure

The orchestration module sits at the top of the hierarchy, importing from all other modules.

The following file structure will be implemented. Files in **bold** are the primary deliverables.

```
mlip_autopipec/
├── app.py                          # The main CLI entry point
├── orchestration/
│   ├── **__init__.py**
│   ├── **workflow_manager.py**     # The centralized State Machine logic
│   ├── **states.py**               # Enum definitions for system states
│   ├── **task_queue.py**           # Abstraction for parallel execution (Dask)
│   ├── **dashboard.py**            # Reporting and visualization logic
│   └── **recovery.py**             # Advanced error handling logic for DFT
└── dft/
    └── **errors.py**               # Definitions of specific DFT failure modes
```

### 2.2. Component Interaction and Data Flow (The Grand Loop)

The system operates as an infinite loop governed by the `WorkflowManager`.

1.  **Initialization**:
    User runs `mlip-auto run config.yaml`.
    -   The Manager initializes the `Database` and `TaskQueue` (Dask Cluster).
    -   It loads the `workflow_state.json` checkpoint. If none exists, it starts at `Generation 0`.

2.  **State 1: COLD_START (Seeding)**:
    -   *Condition*: Database is empty.
    -   *Action*: Call `Generator` to create 1000 structures. Call `Surrogate` to select 50. Submit to `DFT Queue`.
    -   *Transition*: Move to `LABELLING`.

3.  **State 2: LABELLING (The Factory)**:
    -   *Condition*: Pending jobs in `DFT Queue`.
    -   *Action*: Monitor Dask futures.
        -   **Success**: Save result to DB.
        -   **Failure**: Pass stderr to `RecoveryHandler`. If recoverable (e.g., SCF error), create new config and resubmit. If fatal, mark as `FAILED`.
    -   *Transition*: When queue is empty and `n_new_data > threshold`, move to `TRAINING`.

4.  **State 3: TRAINING (The Brain)**:
    -   *Action*: Launch `PacemakerWrapper`.
    -   *Wait*: Block until training completes.
    -   *Result*: New potential `gen_N.yace`.
    -   *Transition*: Move to `INFERENCE`.

5.  **State 4: INFERENCE (The Explorer)**:
    -   *Action*: Launch N parallel `LammpsRunner` tasks using `gen_N.yace`.
    -   *Monitor*: If any task finds high uncertainty, extract the cluster and add to `DFT Queue`.
    -   *Transition*: When batch finishes, if new candidates found, move to `LABELLING`. Else, iterate or stop.

6.  **Reporting**:
    -   Every 5 minutes, `dashboard.generate_report()` queries the DB and updates `index.html`.

## 3. Design Architecture

### 3.1. Workflow Manager (`orchestration/workflow_manager.py`)

-   **Class**: `WorkflowManager`
-   **Attributes**:
    -   `state`: `WorkflowState` (Enum).
    -   `generation`: `int`.
    -   `config`: `GlobalConfig`.
    -   `queue`: `TaskQueue`.
-   **Method**: `tick()`
    -   This is the heartbeat method called in the main loop.
    -   It checks the current state, checks queue status, and executes the state transition logic.
    -   **Persistence**: At the end of every tick, it dumps `state.json` to disk. This ensures that if the process is killed (e.g., power outage), it can resume exactly where it left off.

### 3.2. Task Queue (`orchestration/task_queue.py`)

Abstraction layer for parallelism. We use `dask.distributed` because it handles local parallelism (threads/processes) and distributed clusters (Slurm) with the same API.

-   **Class**: `TaskQueue`
-   **Method**: `submit(func, *args, **kwargs) -> str`
    -   Submits a function to the cluster. Returns a UUID string.
-   **Method**: `get_finished_tasks() -> List[Result]`
    -   Returns results of completed tasks.
-   **Scalability**: By abstracting this, we can swap Dask for Celery or Parsl in the future without changing the workflow logic.

### 3.3. Advanced Recovery (`orchestration/recovery.py`)

This module encodes the "Expert Knowledge" of a computational physicist.

-   **Class**: `RecoveryHandler`
-   **Method**: `analyze_error(stdout: str, stderr: str) -> ErrorType`
    -   Regex matching for common QE errors:
        -   "convergence not achieved" -> `SCF_CONVERGENCE`
        -   "c_bands: n is too large" -> `DIAGONALIZATION_ERROR`
        -   "cholesky decomposition failed" -> `CHOLESKY_ERROR`
-   **Method**: `get_recovery_strategy(error: ErrorType, current_params: Dict) -> Dict`
    -   *Strategy A (Mixing)*: If SCF failed, reduce `mixing_beta` by 30%.
    -   *Strategy B (Algo)*: If Diagonalization failed, switch `diagonalization` from `david` to `cg` (Conjugate Gradient is slower but more robust).
    -   *Strategy C (Temp)*: If Cholesky failed (often due to metallic gap closing), increase `smearing_width` by 0.005 Ry.
    -   *Limit*: Allow max 3 retries per structure.

### 3.4. Dashboard (`orchestration/dashboard.py`)

-   **Class**: `DashboardGenerator`
-   **Library**: `plotly` + `jinja2`.
-   **Method**: `update()`
    -   Queries DB for: RMSE history, Cumulative Data Count, Uncertainty Distribution.
    -   Generates interactive plots.
    -   Writes static HTML file to `output/dashboard/index.html`.

## 4. Implementation Approach

1.  **Phase 1: Recovery Logic (The First Defense)**
    -   Implement `RecoveryHandler`.
    -   Test against a library of "Bad Logs" (we will create mock log files containing specific error strings).
    -   Verify that the handler produces the correct parameter overrides.

2.  **Phase 2: Task Queue (The Engine)**
    -   Set up `dask.distributed.Client` (LocalCluster).
    -   Verify submission and result retrieval with dummy functions (`time.sleep`).

3.  **Phase 3: State Machine (The Brain)**
    -   Implement `WorkflowManager`.
    -   **Grand Mock Test**: This is critical. We will run the entire workflow where *every* component is mocked.
        -   Generator returns random atoms.
        -   DFT returns random energy.
        -   Trainer returns random potential.
        -   Inference returns random candidates.
    -   We verify that the Manager correctly transitions states: Cold Start -> Labelling -> Training -> Inference -> Labelling.

4.  **Phase 4: Integration**
    -   Wire the real components (from Cycles 1-5).
    -   Run the "Mini-Loop" (Scenario 6.1) on a local machine.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Recovery Strategies**:
    -   Input: `mixing_beta=0.7`, Error=`SCF_CONVERGENCE`.
    -   Output: `mixing_beta=0.49`.
    -   Input: `diagonalization='david'`, Error=`DIAGONALIZATION_ERROR`.
    -   Output: `diagonalization='cg'`.
-   **State Persistence**:
    -   Run Manager. Force state `TRAINING`. Save state.
    -   Restart Manager. Assert state is `TRAINING`.

### 5.2. Integration Testing
-   **Dask Resilience**:
    -   Submit a job that raises an Exception.
    -   Verify `TaskQueue` catches it and reports "FAILED" rather than crashing the main process.
-   **Dashboard**:
    -   Populate DB with 100 fake entries.
    -   Run `DashboardGenerator`.
    -   Check that `index.html` exists and is > 0 bytes.

### 5.3. System Testing (The Final Exam)
-   **Zero-Human Run**:
    -   **System**: Pure Aluminum (simple, fast).
    -   **Config**: `mlip-auto run config_al.yaml`.
    -   **Expectation**:
        1.  System generates data.
        2.  Runs DFT (mock or real).
        3.  Trains Gen 0 potential.
        4.  Runs MD.
        5.  Finds high gamma.
        6.  Adds new data.
        7.  Trains Gen 1 potential.
    -   **Pass Criteria**: The loop completes at least 2 full generations without any manual command input.
