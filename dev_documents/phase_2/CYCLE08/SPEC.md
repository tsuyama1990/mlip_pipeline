# Cycle 08 Specification: Orchestration & Production Readiness

## 1. Summary

Cycle 08 is the final cycle. It implements the **Workflow Orchestrator**. Until now, we have built independent modules (Generator, DFT, Trainer, Inference). We have tested them individually. Now we must bind them into a single, cohesive, self-driving application.

This cycle implements the **WorkflowManager**, which acts as the "Brain" of the operation. It decides *what* to do next based on the system state.
-   "Is the DB empty? -> Generate data."
-   "Is data ready? -> Submit DFT."
-   "Is DFT done? -> Train potential."
-   "Is potential ready? -> Run MD."
-   "Did MD fail? -> Extract and loop back."

We use **Dask** for distributed task management. Dask allows us to parallelize the "Heavy" tasks (DFT, MD) across available resources (local cores or SLURM cluster) while keeping the Orchestrator light. We also add a **Dashboard** for real-time observability.

## 2. System Architecture

New components in `src/orchestration`.

```ascii
mlip_autopipec/
├── src/
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── **manager.py**      # The State Machine
│   │   ├── **task_queue.py**   # Dask wrapper
│   │   └── **dashboard.py**    # HTML/JSON reporting
│   └── main.py                 # Final CLI Logic
├── tests/
│   └── orchestration/
│       ├── **test_manager.py**
│       └── **test_dashboard.py**
```

### Key Components

1.  **`WorkflowManager`**: Implements the high-level loop. It persists the `WorkflowState` (e.g., "Current Generation: 2") to disk so it can be stopped and resumed. It holds references to all other modules.
2.  **`TaskQueue`**: Wraps `dask.distributed.Client`. It submits functions like `QERunner.run` to workers and returns Futures. It manages dependencies (e.g., "Don't train until DFT is done"). It handles the distinction between "Fast Tasks" (Surrogate) and "Slow Tasks" (DFT).
3.  **`Dashboard`**: A simple reporter that reads the DB and Logs to produce a `status.html` file with plots (RMSE history, CPU usage, Number of structures).

## 3. Design Architecture

### Domain Concepts

**The "Zero-Human" Loop**:
The loop is infinite (or bounded by `max_generations`).
1.  **Check State**: Resume from checkpoint.
2.  **Phase A (Exploration)**: If no data, call `Generator` + `Surrogate`. Result: Candidates.
3.  **Phase B (Labeling)**: Push candidates to `DFT_Queue`. Wait for completion. Result: Labeled Data.
4.  **Phase C (Learning)**: Train `Pacemaker`. Result: Potential.
5.  **Phase D (exploitation)**: Run `Inference`. Result: Uncertain Structures.
6.  **Loop**: Add Uncertain Structures to `DFT_Queue` and GOTO Phase B.

**Robustness**:
The Manager must handle "Partial Failures". If 10 out of 100 DFT jobs fail, we don't crash. We just train on the 90 successful ones and log the 10 failures. We only crash if the failure rate > 50%.

### Data Models

```python
class WorkflowState(BaseModel):
    current_generation: int = 0
    status: Literal["idle", "dft", "training", "inference"] = "idle"
    pending_tasks: List[str] = []

class OrchestratorConfig(BaseModel):
    max_generations: int = 5
    dask_scheduler_address: Optional[str] = None # None = LocalCluster
    workers: int = 4
```

## 4. Implementation Approach

1.  **Step 1: Dask Integration (`task_queue.py`)**:
    -   Implement `TaskQueue`.
    -   In `__init__`, setup `LocalCluster` or connect to existing one.
    -   Method `submit_dft_batch(structures)`. Use `client.map`.
    -   Method `wait_for_completion(futures)`. Use `dask.distributed.wait`.
2.  **Step 2: State Machine (`manager.py`)**:
    -   Implement `WorkflowManager.run()`.
    -   Use a `while` loop checking `state.current_generation`.
    -   Inside loop, `if/elif` blocks for phases.
    -   Save `state.json` after every major transition.
    -   Implement `resume()` logic that loads `state.json`.
3.  **Step 3: Dashboard (`dashboard.py`)**:
    -   Implement `Dashboard.update()`.
    -   Query ASE-db for "count(id)".
    -   Parse logs for "RMSE".
    -   Use `plotly` or `matplotlib` to save `learning_curve.png`.
    -   Write `index.html` with an auto-refresh meta tag.

## 5. Test Strategy

### Unit Testing Approach (Min 300 words)
-   **State Transitions**: We will unit test the `WorkflowManager` logic by mocking the phases. We will assert that if `status` is "dft" and the queue is empty, it transitions to "training". We will assert that if `current_generation` reaches `max`, it stops. We will verify that state saving writes to disk.
-   **Resumability**: We will create a `state.json` indicating we are in "Generation 3". We will instantiate the Manager and verify it initializes its internal counters to 3, skipping the previous steps. We will test what happens if the state file is corrupted (should raise error or start fresh).
-   **Dashboard Generation**: We will pass a dummy list of metrics (Generation vs RMSE) to the Dashboard. We will assert that it generates a valid HTML file (containing `<html>` tags) and a PNG image. We will check that it doesn't crash if the metrics list is empty.

### Integration Testing Approach (Min 300 words)
-   **The "Grand Mock"**: This is the ultimate test. We will use `pytest-mock` to mock *every* physics engine (`QERunner`, `Pacemaker`, `Lammps`).
    -   **Mock DFT**: Returns random Energy.
    -   **Mock Trainer**: Returns a dummy `.yace` file instantly.
    -   **Mock Inference**: Returns 1 uncertain structure in Gen 0, then 0 uncertain structures in Gen 1.
    -   **Execution**: We run the `WorkflowManager`.
    -   **Expectation**:
        1.  Gen 0: Generates data -> Mock DFT -> Mock Train -> Mock Inference (finds uncertainty).
        2.  Gen 1: Extracts -> Mock DFT -> Mock Train -> Mock Inference (Converged).
        3.  Exit.
    -   This proves the *logic* of the loop is sound, data flows correctly between modules, and the system terminates when criteria are met.
-   **Parallelism**: We will check if `dask` is actually spawning workers. We can do this by checking `client.scheduler_info()['workers']`. We will run a test with `sleep(1)` in the mock DFT function and verify that 4 tasks finish in 1 second (parallel) rather than 4 seconds (serial).
