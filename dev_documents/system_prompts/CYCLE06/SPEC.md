# MLIP-AutoPipe: Cycle 06 Specification

- **Cycle**: 06
- **Title**: Production Readiness - Advanced Embedding and Scalability
- **Status**: Scoping

---

## 1. Summary

Cycle 06 elevates the MLIP-AutoPipe system from a functional prototype to a robust, scalable, and scientifically sophisticated production tool. The objective of this final cycle is twofold: to implement state-of-the-art techniques that significantly improve the physical accuracy of the generated potential, and to re-architect the core workflow for high-throughput execution in a real high-performance computing (HPC) environment. This cycle transitions the system from a serial proof-of-concept to a parallel, resilient, and production-grade scientific instrument.

The first key enhancement is the implementation of a **Periodic Embedding and Force Masking** strategy. The simple active learning loop from Cycle 05 extracts the entire simulation cell when uncertainty is detected. This is inefficient if the uncertainty is caused by a highly localized event (like a defect migration). The new approach will extract a small, fully periodic sub-cell centered on the uncertain atom. This dramatically reduces the cost of the subsequent DFT calculation. More importantly, to avoid unphysical boundary effects in this small cell, we will implement "Force Masking," a technique where the forces on the "buffer" atoms at the edge of the sub-cell are excluded from the training loss function. This ensures the model learns only from the physically correct forces in the core region, vastly improving the quality and transferability of the final potential.

The second part of the cycle focuses on production-readiness. The simple `subprocess` calls to the DFT engine will be replaced with a proper **Task Queue** system (like Dask). This will transform the DFT Factory from a single-file-at-a-time process into a high-throughput engine capable of managing hundreds or thousands of concurrent DFT calculations on a compute cluster. To complement this, the `QEProcessRunner` will be upgraded with a multi-level **automatic error recovery** mechanism. It will be able to intelligently respond to common DFT convergence failures by automatically adjusting calculation parameters and retrying, making the entire workflow resilient to the transient errors common in large-scale computations. Finally, robust checkpointing will be implemented, ensuring that long-running, multi-day workflows can be safely stopped and restarted.

---

## 2. System Architecture

This cycle involves significant refactoring of the main application logic and enhancements to several key modules to support the new, advanced features.

**File Structure for Cycle 06:**

The following files will be created or modified.

```
.
└── src/
    └── mlip_autopipec/
        ├── app.py              # Heavily refactored for task queue integration
        ├── config/
        │   └── system.py       # Modified for new error recovery and embedding params
        ├── data/
        │   └── database.py     # Modified to store force_mask array
        └── modules/
            ├── dft_factory.py  # Upgraded with auto-recovery logic
            ├── trainer.py      # Upgraded to use force_mask data
            └── inference.py    # Upgraded with periodic embedding logic
```

**Component Breakdown:**

*   **`app.py`**: The main application orchestrator will undergo a major change. The simple, serial loop will be replaced with logic that manages a task queue (e.g., a Dask client). Instead of calling `dft_runner.run()` directly, it will now submit that call as a task to the queue and manage the resulting "future" objects, allowing for massively parallel execution.
*   **`modules/inference.py`**: The `LammpsRunner` (or a new utility within this module) will now house the complex `periodic_embedding` logic. When uncertainty is detected, it will no longer yield the full `Atoms` object but will instead perform the sub-cell extraction and yield the smaller, periodic `Atoms` object along with its corresponding `force_mask` array.
*   **`modules/dft_factory.py`**: The `QEProcessRunner`'s `run` method will be wrapped with a new retry mechanism, making it significantly more robust.
*   **`data/database.py`**: The `DatabaseManager`'s `write_calculation` method will be updated to accept the optional `force_mask` array and persist it into the database's key-value pairs.
*   **`modules/trainer.py`**: The `PacemakerTrainer` will be updated to query the database for the `force_mask` and pass this information to the `pacemaker_train` command, which supports per-atom force weighting.

This new architecture represents a shift from a linear pipeline to a parallel, manager-worker model, which is essential for performance at scale.

---

## 3. Design Architecture

The design in Cycle 06 focuses on advanced algorithms and robust, parallel execution patterns.

**Periodic Embedding and Force Masking Design (`inference.py`):**

*   **`_extract_periodic_subcell`**: A new private method will be designed.
    *   **Input**: The full simulation `Atoms` object, the index of the uncertain atom, a cutoff radius `rcut` (from config), and a buffer size `delta_buffer` (from config).
    *   **Algorithm**:
        1.  Define a box size `L = 2 * (rcut + delta_buffer)`.
        2.  Get the position of the uncertain atom `i`.
        3.  Find all atoms within this cubic region centered at `i`, correctly handling atoms that wrap around the periodic boundaries of the *original* simulation cell.
        4.  Create a *new* ASE `Atoms` object containing only these extracted atoms. The cell of this new object will be a cubic box of side length `L`. The positions of the extracted atoms will be remapped into this new cell.
    *   **Output**: The new, smaller, periodic `Atoms` object.
*   **`_generate_force_mask`**: Another new method.
    *   **Input**: The extracted sub-cell `Atoms` object and the cutoff radius `rcut`.
    *   **Algorithm**: It will calculate the distance of each atom in the sub-cell from the center of the box. Atoms with a distance less than `rcut` will have a mask value of `1.0`. Atoms with a distance greater than or equal to `rcut` (i.e., those in the buffer region) will have a mask value of `0.0`.
    *   **Output**: A NumPy array of the same length as the number of atoms in the sub-cell.
*   The `LammpsRunner`'s `run` generator will be updated to `yield` a tuple: `(embedded_atoms, force_mask)`.

**Auto-Recovery and Retry Logic Design (`dft_factory.py`):**

*   The `QEProcessRunner.run` method will be refactored. The core logic will be moved to a private method, e.g., `_perform_calculation`.
*   The public `run` method will now contain a `for` loop, e.g., `for attempt in range(max_retries):`.
*   Inside a `try...except DFTCalculationError:` block, it will call `_perform_calculation`.
*   The `_perform_calculation` method will be passed a (potentially modified) copy of the DFT parameters.
*   In the `except` block, the logic will inspect the error and the `attempt` number. It will modify the DFT parameters for the *next* iteration based on a defined sequence (e.g., on attempt 1, reduce `mixing_beta`; on attempt 2, change `mixing_mode`). If the loop finishes without success, the last exception is re-raised.

**Task Queue Architecture Design (`app.py`):**

*   The application will now require initialization with a Dask client: `client = Client(...)`.
*   A list of pending DFT "futures" will be maintained: `dft_futures = []`.
*   When the `LammpsRunner` yields an uncertain structure, the main loop will *not* block. Instead, it will submit the calculation as a non-blocking task:
    `future = client.submit(dft_runner.run, uncertain_structure)`
    `dft_futures.append(future)`
*   The main loop will then periodically check the status of the futures in the list: `completed_futures = client.gather([f for f in dft_futures if f.done()])`.
*   For each completed and successful calculation, the result is written to the database.
*   The training step will now be triggered periodically or when a certain number of new calculations have been completed, ensuring the trainer works on batches of new data rather than just one point at a time.

---

## 4. Implementation Approach

The implementation will be done in three stages: first the self-contained scientific improvements, then the robustness improvements, and finally the major architectural refactoring for parallelism.

1.  **Implement Embedding and Masking:**
    *   In `inference.py`, write the `_extract_periodic_subcell` helper function, using NumPy for the geometric calculations. Pay careful attention to the periodic boundary condition logic.
    *   Write the `_generate_force_mask` function.
    *   Update the `LammpsRunner`'s `run` generator to call these functions and `yield` the tuple.
    *   In `database.py`, modify `write_calculation` to accept `force_mask=None` and save it to the `key_value_pairs` if it's not `None`.
    *   In `trainer.py`, update the data fetching logic to query for the `force_mask` and prepare it in the format Pacemaker expects for per-atom weighting.

2.  **Implement DFT Auto-Recovery:**
    *   In `dft_factory.py`, refactor the `run` method as described in the design section.
    *   Create a dictionary or list in the config that defines the retry strategy, e.g., `[{'param': 'mixing_beta', 'value': 0.3}, {'param': 'mixing_mode', 'value': 'local-tf'}]`.
    *   The `run` method's `except` block will iterate through this strategy, applying the modifications for each attempt.

3.  **Refactor for Task Queue Parallelism:**
    *   Add `dask` and `distributed` to the `pyproject.toml` dependencies.
    *   In `app.py`, modify the startup logic to initialize a Dask `Client`. This might connect to a local cluster for testing or a remote scheduler in production.
    *   Change the main loop from a simple blocking call to a non-blocking `client.submit` call.
    *   Implement the logic to manage the list of futures, checking for completion and gathering results.
    *   The logic for triggering retraining will need to be updated to fire, for example, after every `N` new calculations are completed.

4.  **Implement Checkpointing:**
    *   All key states (the list of pending future IDs, the path to the current model, the state of the MD simulation) will be periodically saved to a checkpoint file (e.g., a JSON file).
    *   The application startup logic in `app.py` will be modified to check for the existence of this checkpoint file and, if found, restore its state from it, allowing the workflow to resume.

---

## 5. Test Strategy

Testing for this final cycle is focused on the new advanced algorithms and the complex asynchronous nature of the refactored application.

**Unit Testing Approach (Min 300 words):**

Unit tests will be critical for verifying the correctness of the complex, self-contained algorithms introduced in this cycle.

*   **Periodic Embedding Algorithm (`tests/modules/test_inference.py`):**
    A test function, `test_periodic_embedding_logic`, will be created to rigorously validate the extraction logic. We will construct a large (e.g., 5x5x5) pristine FCC crystal `Atoms` object. We will then choose an atom deep inside the crystal as the "uncertain" atom. We will call the `_extract_periodic_subcell` function with a specific cutoff and buffer. The test will perform several assertions on the returned sub-cell object. First, it will assert that the number of atoms in the sub-cell is correct based on the known density of the crystal and the volume of the extraction box. Second, it will assert that the new cell is cubic and has the correct dimensions (`L x L x L`). Third, and most critically, we will test the periodic boundary handling. We will choose an "uncertain" atom near the corner of the original cell and assert that the extracted sub-cell correctly includes atoms from the other side of the periodic boundary, verifying that our wrapping logic is correct.

*   **Force Mask Generation (`tests/modules/test_inference.py`):**
    A dedicated test, `test_force_masking`, will verify the mask creation. Using a pre-defined sub-cell, it will call `_generate_force_mask`. It will then iterate through the atoms of the sub-cell and the generated mask, asserting that atoms inside the cutoff radius have a mask value of 1.0 and atoms outside have a value of 0.0.

*   **DFT Retry Logic (`tests/modules/test_dft_factory.py`):**
    A test, `test_qe_runner_retry_logic`, will mock `subprocess.run` to throw an exception multiple times before finally succeeding. We will use `mocker.spy` to track the arguments passed to the private `_generate_input_file` method. The test will assert that after the first failure, the method is called again with modified parameters (e.g., a lower `mixing_beta`). It will assert this for the entire chain of recovery steps, confirming the state machine is working as designed.

**Integration Testing Approach (Min 300 words):**

Integration testing will focus on the new asynchronous workflow managed by the task queue.

*   **Dask Task Queue Workflow (`tests/test_app.py`):**
    The main integration test, `test_app_with_dask_local_cluster`, will verify the parallel execution logic.
    1.  **`LocalCluster`**: The test will programmatically start a `dask.distributed.LocalCluster` and `Client` in the test setup. This creates a real, albeit local, manager-worker environment.
    2.  **Mocks**: The `QEProcessRunner.run` method itself will be mocked, but this time the mock will be a simple function that, for example, sleeps for a short duration and returns a dummy result. This is so we are testing the Dask scheduling, not the DFT logic.
    3.  **Execution**: The test will invoke the main application logic, which will be configured to connect to our local Dask cluster. The mocked `LammpsRunner` will be configured to yield two uncertain structures in quick succession.
    4.  **Assertions**: The application logic will submit two tasks to the Dask cluster. We will not assert the final result directly. Instead, we will assert that the `client.submit` method was called twice. We will then wait for the futures to complete and assert that the `db_manager.write_calculation` method was subsequently called twice. This confirms that the application can correctly submit jobs to a task queue in a non-blocking way and gather the results asynchronously. This test validates the core architectural change from a serial to a parallel workflow.
