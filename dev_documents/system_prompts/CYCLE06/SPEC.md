# CYCLE06: Resilience and Scalability (SPEC.md)

## 1. Summary

This document provides the detailed technical specification for Cycle 6 of the MLIP-AutoPipe project. Having established the core functional components of the active learning loop in previous cycles, this cycle addresses the critical non-functional requirements that make the system viable for real-world, long-duration scientific campaigns: **resilience and scalability**. A "Zero-Human" protocol is only truly autonomous if it can withstand the inevitable failures of a complex, distributed computing environment and can effectively leverage the power of modern parallel hardware.

The first key deliverable is a robust **checkpointing and recovery system**. Long-running workflows, which can last for days or weeks, are highly vulnerable to interruptions such as HPC node failures, scheduler maintenance, or power outages. This cycle will introduce a `CheckpointState` data model and logic within the `WorkflowManager` to periodically save the entire state of the workflow to a file. This state includes the list of all pending and completed DFT calculations, the path to the current best potential, and the active learning generation number. If the system is stopped, it can be restarted and will use this checkpoint file to seamlessly resume exactly where it left off, preventing the loss of valuable computation time.

The second key deliverable is **scalability through parallel execution**. A typical MLIP generation campaign requires thousands of independent DFT calculations. Executing these serially would be impractically slow. This cycle will integrate the Dask distributed computing library to manage a task queue. The `WorkflowManager` will be refactored to act as a central dispatcher, submitting hundreds of DFT calculation jobs to a Dask cluster as non-blocking, parallel tasks. It will then manage the `Future` objects returned by Dask, processing results as they become available. This architecture transforms the system from a linear process into a high-throughput factory, allowing it to take full advantage of HPC clusters and dramatically reduce the time-to-solution. This cycle will also formalise the DFT auto-recovery logic into a generic, reusable decorator for enhanced code clarity and robustness.

## 2. System Architecture

The architecture for Cycle 6 primarily involves significant enhancements to the `WorkflowManager` and the introduction of new utility modules for handling resilience and Dask integration.

**File Structure for Cycle 6:**

The following ASCII tree highlights the new or heavily modified files in **bold**.

```
mlip-autopipe/
├── dev_documents/
│   └── system_prompts/
│       └── CYCLE06/
│           ├── **SPEC.md**
│           └── **UAT.md**
├── mlip_autopipec/
│   ├── **workflow_manager.py** # Major refactor for Dask and checkpointing
│   ├── config/
│   │   └── models.py           # Add CheckpointState model
│   ├── modules/
│   │   └── dft.py              # Refactor to use @retry decorator
│   └── utils/
│       ├── __init__.py
│       ├── ase_utils.py
│       ├── **dask_utils.py**   # Helpers for Dask client setup
│       └── **resilience.py**   # Generic @retry decorator
├── tests/
│   ├── **test_workflow_manager.py** # New tests for checkpointing/resuming
│   └── utils/
│       └── **test_resilience.py**   # Tests for the retry decorator
└── pyproject.toml
```

**Component Blueprint: `workflow_manager.py` (Refactored)**

The `WorkflowManager` evolves from a conceptual orchestrator to a stateful, resilient, and parallel dispatcher.

-   **`__init__`**: Now checks for the existence of a checkpoint file in the working directory. If found, it calls `_load_checkpoint`.
-   **`run()`**: The main loop is completely redesigned. Instead of a simple `for` loop, it becomes a persistent loop that manages a collection of Dask `Future` objects. It will use `dask.distributed.as_completed` to process results as they finish.
-   **`_submit_dft_batch(self, structures: List[ase.Atoms])`**: A new method that takes a list of structures, wraps them in `DFTJob` objects, and submits them to the Dask cluster using `dask_client.submit()`. It stores the returned `Future` objects.
-   **`_save_checkpoint(self)`**: Serialises the current `CheckpointState` Pydantic model to a `checkpoint.json` file. This will be called after every significant state change (e.g., after submitting a new batch of jobs, after a new model is trained).
-   **`_load_checkpoint(self)`**: Reads the `checkpoint.json` file and deserialises it into the `WorkflowManager`'s state, enabling a seamless resume.

**Component Blueprint: `utils/resilience.py` (New)**

This module provides generic, reusable patterns for building robust functions.

-   **`@retry(attempts: int, delay: float, exceptions: Tuple[Type[Exception], ...])`**: A decorator that can be applied to any function. It will wrap the function call in a `try...except` block. If one of the specified `exceptions` is caught, it will wait for `delay` seconds and retry the call, up to `attempts` times. This formalises the auto-recovery logic.

**Component Blueprint: `modules/dft.py` (Refactored)**

-   The `run` method in `DFTFactory` will be refactored. The complex, manual `for` loop for retries will be removed. Instead, the method will be decorated with `@retry(attempts=3, delay=5, exceptions=(DFTCalculationError,))`. The core of the method will be simplified to a single attempt, with the decorator handling the retry logic externally.

**Component Blueprint: `utils/dask_utils.py` (New)**

-   **`get_dask_client()`**: A helper function to connect to a Dask cluster. It will read the scheduler address from an environment variable or a configuration file, making it easy to switch between a local test cluster and a real HPC Dask deployment.

## 3. Design Architecture

This cycle's design focuses on making the workflow's state explicitly serializable and its execution distributable.

**Pydantic Schema Definitions (in `config/models.py`):**

A new model is introduced to represent the entire state of the workflow at any given time.

-   **`CheckpointState(BaseModel)`**:
    -   `run_uuid: UUID`: The unique ID for the entire workflow run.
    -   `system_config: SystemConfig`: A copy of the full system configuration.
    -   `active_learning_generation: int`: The current generation number (e.g., 0 for the initial set, 1 after the first retraining).
    -   `current_potential_path: Optional[Path]`: The path to the latest `.yace` file.
    -   `pending_job_ids: List[UUID]`: A list of UUIDs for jobs that have been submitted to Dask but whose results have not yet been processed and saved to the database.
    -   `job_submission_args: Dict[UUID, Any]`: A dictionary mapping a job's UUID to the arguments needed to resubmit it (e.g., the `ase.Atoms` object and its metadata). This is critical: we do not save the Dask `Future` itself, but the *arguments* required to recreate the task.

**Data Flow and Logic:**

**Checkpointing/Resuming Workflow:**
1.  **Start:** `WorkflowManager` is instantiated. It looks for `checkpoint.json`.
2.  **Resume:** If the file exists, the manager calls `_load_checkpoint`. It populates its internal state from the file. It then iterates through the `pending_job_ids` and uses the `job_submission_args` to re-submit all the lost jobs to the Dask cluster. The workflow is now back in the exact state it was in before the interruption.
3.  **Run:** The manager enters its main `as_completed` loop, processing results.
4.  **State Change:** A batch of new structures is ready to be submitted.
5.  **Save State:** Before calling `dask_client.submit()`, the manager adds the new jobs' arguments to `job_submission_args` and their IDs to `pending_job_ids`. It then calls `_save_checkpoint()`, writing the new state to disk.
6.  **Submit:** The jobs are submitted to Dask.
7.  **Process Result:** A DFT calculation finishes. The manager gets the result from the `as_completed` iterator.
8.  **Save Result & State:** The result is saved to the database. The corresponding job ID is removed from `pending_job_ids` and `job_submission_args`. The manager calls `_save_checkpoint()` again to reflect that this job is now complete.
9.  This "save-then-act" pattern ensures that the checkpoint file is always an accurate reflection of the work that is currently in progress.

**Dask Parallel Execution:**
-   The `WorkflowManager` is the single point of contact with the Dask cluster.
-   The `DFTFactory.run` method is now a pure, stateless function that takes an `ase.Atoms` object and returns a `DFTResult`.
-   The `WorkflowManager` submits calls to this function via `dask_client.submit(dft_factory.run, atoms_object)`.
-   This returns a `Future` immediately. The manager does not block. It collects many such futures.
-   The main program loop iterates over `dask.distributed.as_completed(list_of_futures)`, which yields results as they become available, regardless of the order they were submitted. This ensures maximum efficiency, as the system is always ready to process the next available result.

## 4. Implementation Approach

1.  **Implement `@retry` Decorator:** Create the `utils/resilience.py` module and implement the generic `@retry` decorator. Write unit tests for it in `tests/utils/test_resilience.py`.
2.  **Refactor `DFTFactory`:** Modify `modules/dft.py` to remove the manual retry loop and apply the new decorator to the `run` method.
3.  **Implement Dask Utilities:** Create the `utils/dask_utils.py` module and the `get_dask_client` function.
4.  **Define `CheckpointState` Model:** Add the `CheckpointState` Pydantic model to `config/models.py`.
5.  **Implement Checkpointing in `WorkflowManager`:** Add the `_save_checkpoint` and `_load_checkpoint` methods to `workflow_manager.py`. These methods will handle the JSON serialization/deserialization of the `CheckpointState` model.
6.  **Refactor `WorkflowManager.run` for Dask:** This is the most significant step. The existing `run` method will be replaced with a new Dask-centric loop.
    -   It will initialize the Dask client.
    -   It will manage a list of submitted `Future` objects.
    -   The core of the method will be a `while` loop that continues as long as there are active or pending tasks.
    -   Inside the loop, `as_completed` will be used to process results.
7.  **Integrate Checkpointing with the Dask Loop:** Weave calls to `_save_checkpoint` into the new Dask loop, implementing the "save-then-act" logic described in the design section.
8.  **Implement Resume Logic:** Add the logic to the `__init__` method of the `WorkflowManager` to load a checkpoint and re-submit pending jobs if a checkpoint file is found.

## 5. Test Strategy

Testing this cycle is crucial for ensuring the system is truly robust. It requires testing failure modes and the interaction with a real, albeit local, parallel execution system.

**Unit Testing Approach (Min 300 words):**

-   **Testing `@retry`:** The decorator will be tested with a mock function. `test_retry_succeeds_on_first_try` will use a function that never fails and assert it's called only once. `test_retry_succeeds_after_failures` will use `unittest.mock` to make a function that raises an exception twice and then succeeds on the third call; the test will assert the decorator handles this and the function is called three times. `test_retry_fails_after_max_attempts` will test a function that always fails, asserting that the decorator gives up and re-raises the exception after the maximum number of attempts.
-   **Testing Checkpointing:** `test_save_and_load_checkpoint_are_symmetric` will create a complex `CheckpointState` object, call `_save_checkpoint`, then immediately call `_load_checkpoint` from that file, and assert that every field of the loaded state is identical to the original. This verifies that the serialization-deserialization process is lossless.
-   **Mocking Dask:** We can test the `WorkflowManager`'s submission logic without a real Dask cluster by patching the `dask_client` object with a `MagicMock`. We can then assert that `mock_client.submit` is called with the correct arguments and that the number of calls matches the number of structures we intended to submit.

**Integration Testing Approach (Min 300 words):**

Integration tests will use a `dask.distributed.LocalCluster` to create a real multi-worker Dask environment for the duration of the test.

-   **The "Crash and Resume" Test:** This is the most critical test for this cycle.
    1.  **Setup:** Configure a workflow to run, say, 20 DFT calculations.
    2.  **Run Part 1:** Launch the `WorkflowManager` in a separate process. Let it run long enough to submit all 20 jobs but not long enough for them to finish. Then, forcefully terminate the process (`process.kill()`).
    3.  **Verify Checkpoint:** In the main test process, open the `checkpoint.json` file. Assert that it exists and that its `pending_job_ids` list contains 20 UUIDs.
    4.  **Run Part 2:** Instantiate a *new* `WorkflowManager` in a new process, pointing to the same working directory.
    5.  **Verify Resume:** Monitor the logs or the Dask dashboard to confirm that the new `WorkflowManager` loads the state and re-submits the 20 jobs. Let the run complete.
    6.  **Final Assertion:** Check the final database and assert that it contains the results for all 20 calculations, proving that the system successfully recovered from a catastrophic failure with no loss of work.

-   **Parallel Execution Speedup Test:** This test will verify that Dask is actually running jobs in parallel.
    1.  Create a mock `dft_factory.run` function that simply does `time.sleep(2)` and returns a dummy result.
    2.  Run a workflow that submits 8 of these mock jobs to a `LocalCluster` with 4 workers.
    3.  Measure the total wall-clock time for the entire batch to complete.
    4.  Assert that the total time is significantly less than 16 seconds (8 jobs * 2s/job). It should be closer to 4-5 seconds (2 rounds of 4 parallel jobs, plus overhead). This provides definitive proof of parallel execution.
