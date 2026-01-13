# CYCLE03 Specification: The Intelligence

## 1. Summary

This document provides the detailed technical specification for CYCLE03 of the MLIP-AutoPipe project. This cycle marks a pivotal transition, elevating the system from a semi-automated "apprentice" to a fully autonomous, intelligent agent. The core objective of CYCLE03 is to close the active learning loop. This will be achieved by implementing **Module E, the Scalable Inference & On-the-Fly (OTF) Engine**, and by automating the connection between this engine, the DFT factory, and the trainer. By the end of this cycle, the system will be capable of starting with a basic potential, running a simulation, intelligently identifying regions where the potential is uncertain, automatically gathering new DFT data to patch these weaknesses, and retraining itself—all without any human intervention.

**Module E (Scalable Inference & OTF Engine)** will be the workhorse of the production simulation. It will use the best available Machine Learning Interatomic Potential (MLIP) to run large-scale, long-duration molecular dynamics (MD) simulations using the LAMMPS engine. The key feature of this module is its ability to perform "inference with introspection." While the simulation is running, Module E will continuously monitor the uncertainty of the potential for the current atomic configuration. This is achieved by calculating the `extrapolation_grade` provided by the Pacemaker library, a metric that quantifies how much a given structure deviates from the existing training data.

When this uncertainty metric exceeds a predefined threshold, the OTF engine will trigger a crucial new workflow: **Periodic Embedding and Force Masking**. Instead of just saving the single uncertain atomic configuration, the system will extract a small, periodic sub-system centered on the region of high uncertainty. This "embedding" process is superior to traditional cluster-based extraction as it preserves the periodic nature of the bulk material. Furthermore, to avoid learning artifacts from the artificial boundaries of this new small cell, a "force mask" will be applied to the atoms in a buffer region near the boundary, ensuring that only the forces on the core, bulk-like atoms are used for training.

Finally, this cycle will fully automate the training loop. The extracted and masked structures will be fed into a priority queue which is consumed by the DFT Factory (Module C). The Trainer (Module D) will be reconfigured to run as a daemon process, continuously monitoring the central database for the arrival of new data. Once a sufficient amount of new data is available, it will automatically initiate a retraining job, producing an improved, versioned potential. The OTF engine will then be able to hot-swap this new potential to continue its simulation with ever-increasing accuracy and reliability.

## 2. System Architecture

The architecture for CYCLE03 is a significant evolution, transforming the linear pipeline of the previous cycles into a continuous, closed loop. Module E becomes the central hub of activity, driving the simulation and feeding new data back into the training pipeline (Modules C and D), which now run concurrently as background services.

**File Structure for CYCLE03:**

The file structure is expanded to include the new inference module and the necessary logic for task queuing and process management.

```
mlip_autopipec/
├── src/
│   ├── mlip_autopipec/
│   │   ├── __init__.py
│   │   ├── main.py             # CLI updated with 'otf' run command
│   │   ├── settings.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── user_config.py    # Updated with OTF parameters
│   │   │   ├── system_config.py  # Updated with Inference and Queue sections
│   │   │   ├── dft.py
│   │   │   └── data.py           # Schema updated with uncertainty/mask fields
│   │   ├── modules/
│   │   │   ├── __init__.py
│   │   │   ├── a_generator.py
│   │   │   ├── b_explorer.py
│   │   │   ├── c_dft_factory.py  # Refactored to run as a worker/daemon
│   │   │   ├── d_trainer.py      # Refactored to run as a worker/daemon
│   │   │   └── **e_inference.py**    # Module E: Scalable Inference & OTF
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── ase_utils.py
│   │       ├── qe_utils.py
│   │       ├── pacemaker_utils.py
│   │       └── **lammps_utils.py**   # Utilities for LAMMPS
├── tests/
│   ├── __init__.py
│   ├── test_schemas.py
│   └── test_modules/
│       ├── __init__.py
│       ├── ...
│       └── **test_e_inference.py**
├── pyproject.toml
└── README.md
```

**Architectural Blueprint:**

The new architecture is best described as a multi-process system orchestrated by a central task queue.

1.  **Initiation**: The user starts an "OTF" run via the CLI. The system first ensures an initial potential exists (either user-provided or trained from a CYCLE02-style initial dataset).
2.  **Daemonization**: The **DFT Factory (Module C)** and the **Trainer (Module D)** are launched as long-running, background worker processes. They connect to a central, persistent task queue (e.g., implemented with Redis or a simple file-based queue for now).
    *   The DFT Factory worker continuously pulls new structures from the "DFT Queue", calculates their properties, and saves the results to the database.
    *   The Trainer worker periodically checks the database for new entries. When a certain number of new entries are found, it pulls the entire dataset, trains a new potential, and places the new `.yace` file in a designated, versioned directory.
3.  **Main Loop (Module E)**: The **Inference Engine (`e_inference.py`)** starts its main loop.
    *   It loads the latest available potential.
    *   It begins a LAMMPS MD simulation.
    *   At regular intervals (e.g., every 10-100 steps), it uses Pacemaker's library functions to calculate the `extrapolation_grade` for the current atomic configuration.
    *   **Decision Point**: If `extrapolation_grade` is below the threshold, the simulation continues.
    *   **Trigger**: If `extrapolation_grade` exceeds the threshold, the simulation is paused.
4.  **Embedding Workflow**:
    *   The current, high-uncertainty `ase.Atoms` object is passed to the **Periodic Embedding** utility.
    *   This utility identifies the atom with the highest uncertainty and carves out a small periodic box (e.g., `2 * (cutoff + buffer)`) around it from the main simulation cell.
    *   The **Force Masking** logic is applied. It identifies atoms within the `buffer` region of this new, small cell and stores a boolean mask (or a list of indices) as metadata.
    *   This new, smaller `ase.Atoms` object, along with its metadata (including the force mask), is placed onto the "DFT Queue".
5.  **Loop Continuation**: The Inference Engine can then decide to either terminate, or, more advancedly, rewind the simulation slightly and restart with a smaller timestep or with the newly trained potential once it becomes available. The main loop continues.
6.  **Hot-Swapping**: The Inference Engine periodically checks for new, better potentials in the output directory. If a new one is found, it can be "hot-swapped" by restarting the LAMMPS simulation with the updated model.

This closed-loop, asynchronous architecture is the core of the system's intelligence, allowing it to learn and adapt in response to the simulations it is performing.

## 3. Design Architecture

The design for CYCLE03 focuses on robust inter-process communication, state management, and the implementation of the core OTF and embedding algorithms.

**Pydantic Schema Design:**

*   **`system_config.py`**: The main configuration will be expanded.
    *   `OTFConfig(BaseModel)`: This will define the parameters for the OTF loop, such as `uncertainty_threshold: float`, `check_frequency: int` (how many MD steps between checks), and `max_extractions: int`.
    *   `EmbeddingConfig(BaseModel)`: This will define the parameters for the embedding process, including `cutoff: float` and `buffer_size: float`.
    *   `QueueConfig(BaseModel)`: Specifies the type and connection details for the task queue (e.g., path to a file-based queue, Redis URL).

*   **`data.py`**: The database record schema will be updated to support the new workflow.
    *   `StructureRecord(BaseModel)`: Will be extended to include `uncertainty_gamma: Optional[float]` and `force_mask: Optional[List[int]]` (a list of atomic indices to mask during training).
    *   **Producers**: Module E is now a major producer of data records for the queue.
    *   **Consumers**: Module D (Trainer) must now be able to read and interpret the `force_mask` field, passing this information to Pacemaker during training.

**Module and Utility Design:**

*   **Task Queue**: While a full-blown Celery/Dask implementation is deferred to a later cycle, a robust file-based queue will be implemented first. This can be a simple directory where each task is a file with a unique ID, and workers move files between "pending", "running", and "done" subdirectories. This is resilient and simple to implement and debug.

*   **Module E (`e_inference.py`):**
    *   The core will be an `OTFRunner` class.
    *   `__init__`: Will take the `OTFConfig` and `EmbeddingConfig`, and initialize the LAMMPS calculator with the starting potential.
    *   `run()`: The main loop described in the blueprint. It will use Python's `multiprocessing` or `concurrent.futures` to launch the DFT and Trainer workers as background processes.
    *   `_check_uncertainty()`: A method that gets the current `ase.Atoms` from the running simulation and uses a Pacemaker utility function to calculate the `extrapolation_grade`.
    *   `_perform_embedding()`: This method will contain the critical logic for carving out the periodic sub-cell and generating the force mask. This will require careful geometric calculations using NumPy and ASE's cell manipulation functions.

*   **Module C & D (`c_dft_factory.py`, `d_trainer.py`) Refactoring:**
    *   Each module will be refactored into a `Worker` class (e.g., `DFTWorker`).
    *   Each worker will have a `start()` method that begins a continuous loop: `while True: task = self.queue.get(); self.process(task);`.
    *   The core logic (running QE, running Pacemaker) will be moved into the `process()` method.
    *   The Trainer's `process()` method will be updated to look for the `force_mask` in the data and configure the Pacemaker input accordingly to set the weights of the masked atoms' forces to zero.

*   **`lammps_utils.py`**: This new utility file will contain helper functions to set up and run LAMMPS simulations using `ase.calculators.lammpslib`. It will abstract away the complexities of writing LAMMPS input scripts and managing the LAMMPS process.

This design emphasizes robustness and asynchronicity. The file-based queue ensures that even if one component crashes, the overall state of the system is preserved and can be resumed. The clear separation of the main OTF loop from the background workers simplifies the logic of each component.

## 4. Implementation Approach

The implementation will be tackled in a logical sequence: first the core algorithms, then the background workers, and finally the main OTF loop that ties them all together.

**Step 1: Implement Core Algorithms and Utilities**
*   Create `utils/lammps_utils.py` and implement the basic functions for running an MD simulation with an ASE/LAMMPS calculator.
*   In `utils/ase_utils.py` or a new `embedding_utils.py`, implement the `extract_periodic_subsystem` function. This is a purely algorithmic task. The input will be a large `ase.Atoms` object, an atomic index, a cutoff, and a buffer size. The output will be a new, smaller `ase.Atoms` object and the list of masked atomic indices. This should be developed with its own set of unit tests.
*   In `utils/pacemaker_utils.py`, add a function `get_uncertainty(atoms, potential)` that takes an `ase.Atoms` object and a loaded Pacemaker potential, and returns the `extrapolation_grade`.

**Step 2: Refactor Modules into Workers**
*   Implement the file-based task queue system. This will be a simple class that manages tasks in a directory structure.
*   Refactor `c_dft_factory.py` into a `DFTWorker` class with the `start()` and `process()` methods. The `process` method will contain the existing logic for running a single DFT calculation.
*   Refactor `d_trainer.py` into a `TrainerWorker`. Its `process` method will be simpler; it doesn't take a specific task but is triggered by the amount of new data. It needs to be modified to handle the `force_mask` when preparing data for Pacemaker. This involves setting the `weights` for the masked forces to 0 in the training dataset.

**Step 3: Implement the OTF Engine (Module E)**
*   Create `modules/e_inference.py` with the `OTFRunner` class.
*   Implement the `run` method. This will be the main orchestrator. It will start by launching the `DFTWorker` and `TrainerWorker` in separate processes.
*   The main OTF loop will be implemented here. It will initialize the LAMMPS simulation.
*   In a `for` loop over the simulation steps, it will call `md.run(steps)`. After each chunk of steps, it will call the `_check_uncertainty` method.
*   If uncertainty is high, it will call `_perform_embedding` and put the resulting structure onto the DFT task queue.

**Step 4: Update the CLI**
*   Modify `main.py` to add a new `otf` command.
*   This command will be responsible for setting up the initial state (directories, queues, initial potential) and then creating and starting the `OTFRunner`.

This phased approach allows us to build and test the most complex new pieces (the embedding algorithm, the worker framework) in isolation before integrating them into the final, complex, multi-process OTF loop.

## 5. Test Strategy

Testing CYCLE03 is exceptionally challenging due to its asynchronous, multi-process, and non-deterministic nature. The strategy will rely heavily on targeted integration tests that validate the key feedback loop, supplemented by rigorous unit tests for the new, complex algorithms.

**Unit Testing Approach (Min 300 words):**

Unit tests are our first line of defense and are essential for verifying the complex new algorithms in a controlled environment.

*   **Periodic Embedding (`test_embedding_utils.py`):** The periodic embedding and force masking logic is purely algorithmic and highly prone to off-by-one errors or incorrect boundary handling. We will test this function exhaustively.
    *   We will create a large, simple cubic `ase.Atoms` object (e.g., a 10x10x10 grid of Argon atoms).
    *   We will call `extract_periodic_subsystem` for an atom near the center of the large cell. We will then assert that the new, smaller cell has the correct, expected dimensions (`2 * (cutoff + buffer)`). We will assert that the number of atoms in the new cell is correct.
    *   We will check the force mask. We will calculate which atoms *should* be in the buffer region based on their distance from the center of the new cell and assert that the returned list of masked indices matches our calculation exactly.
    *   We will test edge cases, such as when the extraction region wraps around the periodic boundaries of the original cell, ensuring the function correctly handles atoms from opposite sides of the simulation box.

*   **Task Queue (`test_queue.py`):** The file-based task queue will be tested to ensure it is robust to race conditions (as much as possible in a testing environment) and correctly manages task state. We will simulate multiple worker processes trying to access the queue simultaneously, ensuring that a task is only ever assigned to one worker.

*   **Worker Modules (`test_c_dft_factory.py`, `test_d_trainer.py`):** We will unit-test the refactored worker classes by mocking the queue. We will place a mock task on the queue, call the worker's `process` method directly, and assert that it performs the correct action. For the `TrainerWorker`, we will create a test database containing a record with a `force_mask` and assert that the generated Pacemaker input file correctly sets the weights for the specified forces to zero.

**Integration Testing Approach (Min 300 words):**

The integration test for CYCLE03 is the most critical test of the entire project so far. Its goal is to verify that the entire active learning feedback loop functions correctly, from detecting uncertainty to retraining a better potential.

*   **Test Scenario: "Healing" a Defective 2D Lennard-Jones Potential**
    *   **Objective**: To create a deliberately flawed MLIP, run a simulation that exposes its flaws, and verify that the OTF pipeline automatically identifies the weakness, gathers new data, and retrains a "healed" potential that fixes the flaw. A 2D Lennard-Jones (LJ) system is perfect for this as it's computationally trivial, and its physics is simple to visualize and validate.
    *   **Setup:**
        1.  **Create a "defective" potential**: We will generate a training set for a 2D LJ liquid but will deliberately exclude any data for atoms at very short distances (high compression). We will then train an initial MLIP (`lj_defective.yace`) on this incomplete dataset. This potential will be accurate for the liquid state but will behave nonsensically when two atoms get too close.
        2.  **Initial State**: The test will start with this `lj_defective.yace` as the input potential.
    *   **Execution:**
        1.  We will launch the full OTF pipeline using the `otf` command.
        2.  The `OTFRunner` will start an MD simulation of the 2D LJ system. To guarantee we hit the flaw, the simulation will be an NVE simulation started with high initial velocities, ensuring atom pairs will collide.
        3.  The mock "DFT Factory" for this test will not run QE but will instead calculate the exact analytical LJ energy and forces for any submitted structure. This makes the test fast and perfectly accurate.
    *   **Validation:** We will monitor the system's behavior and assert the following sequence of events:
        1.  The MD simulation will start. As two atoms approach a close-contact collision, the `extrapolation_grade` from the defective potential will spike, as it has never seen this configuration. We will assert that this uncertainty is detected.
        2.  We will assert that the simulation pauses and the embedding logic is triggered. A new structure representing the colliding pair will be sent to the "DFT Queue".
        3.  We will assert that our mock DFT Factory processes this structure and adds the correct analytical LJ data to the database.
        4.  We will assert that the `TrainerWorker` detects this new data and automatically starts a retraining process.
        5.  A new potential, `lj_healed_v2.yace`, must be generated.
        6.  **Final Proof**: We will load this new, healed potential. We will use it to run a new MD simulation of the same high-velocity collision. We will assert that this time, the simulation is stable and does *not* trigger the uncertainty mechanism, because the potential now has the correct data for this scenario. This successful, stable run provides definitive proof that the entire feedback loop is working as intended.
