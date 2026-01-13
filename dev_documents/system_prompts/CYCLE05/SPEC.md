# CYCLE05 Specification: The Power-Up

## 1. Summary

This document provides the detailed technical specification for CYCLE05 of the MLIP-AutoPipe project. With the core autonomous workflow and a polished user interface now in place, this cycle focuses on "powering up" the system's capabilities. The objectives are twofold: first, to significantly expand the range of scientific problems the system can solve by implementing **Advanced Simulation Goals**, and second, to enhance the system's robustness and scalability for large-scale production runs through **Performance and Scalability Optimizations**.

The main feature of this cycle is the expansion of the Heuristic Engine and the backend workflows to support more complex simulation types beyond simple equilibration. This will include, at a minimum, workflows for:
1.  **Phase Diagram Construction**: Automatically sampling different compositions and crystal structures to map out the phase stability of binary or ternary alloys.
2.  **Diffusion Coefficient Calculation**: Running long-duration Molecular Dynamics (MD) simulations at various temperatures and analyzing the resulting trajectories to compute diffusion coefficients via Mean Squared Displacement (MSD).
3.  **Elastic Constant Determination**: Applying a series of specific lattice deformations to a crystal to calculate the full elastic tensor ($C_{ij}$), a key predictor of mechanical properties.

Implementing these goals requires not just new analysis scripts, but a deeper integration into the Heuristic Engine and the OTF loop, as each goal has its own unique requirements for structure generation, simulation parameters, and active learning triggers.

The second major part of this cycle is to move beyond the simple, file-based task queue and worker management implemented in CYCLE03. To handle the thousands of concurrent tasks required for things like phase diagram construction, the system will be upgraded to use a production-grade, distributed task queueing system. **Celery**, with a **Redis** broker, will be integrated to manage the DFT calculations and training jobs. This will provide better scalability, resilience, and introspection into the state of the task queues, making the entire system more robust and suitable for deployment on large High-Performance Computing (HPC) clusters.

By the end of CYCLE05, MLIP-AutoPipe will have evolved from a tool for building potentials into a comprehensive platform for automated materials property prediction, capable of tackling complex scientific questions at scale.

## 2. System Architecture

The architecture in CYCLE05 sees a significant upgrade in the task management infrastructure and a functional expansion of the Heuristic Engine and the inference/analysis modules.

**File Structure for CYCLE05:**

The file structure will be refactored to replace the simple queue with a Celery-based implementation. New modules for specific scientific analyses will also be added.

```
mlip_autopipec/
├── src/
│   ├── mlip_autopipec/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── heuristic_engine.py # Significantly expanded
│   │   ├── main.py
│   │   ├── settings.py
│   │   ├── **celery_app.py**       # Celery application definition and configuration
│   │   ├── **tasks.py**            # Definitions of Celery tasks (e.g., run_dft, run_training)
│   │   ├── schemas/
│   │   │   └── ...
│   │   ├── modules/
│   │   │   ├── ...
│   │   │   ├── c_dft_factory.py  # Refactored as a Celery task
│   │   │   ├── d_trainer.py      # Refactored as a Celery task
│   │   │   └── e_inference.py    # Logic updated to handle different simulation goals
│   │   ├── **analysis/**
│   │   │   ├── __init__.py
│   │   │   ├── **diffusion.py**    # MSD analysis logic
│   │   │   ├── **elastic.py**      # Elastic constant fitting logic
│   │   │   └── **phase_diagram.py**# Convex hull analysis logic
│   │   ├── dashboard/
│   │   │   └── ...
│   │   └── utils/
│   │       └── ...
├── tests/
│   └── ...
│   └── **test_analysis/**
│       ├── **test_diffusion.py**
│       └── **test_elastic.py**
├── pyproject.toml
└── README.md
```

**Architectural Blueprint:**

1.  **Task Management Overhaul**:
    *   The simple file-based queue and `multiprocessing` workers from CYCLE03 are completely removed.
    *   A new file, `celery_app.py`, is created to define and configure the central Celery application. It will be configured to use a Redis server as its message broker and results backend.
    *   The core logic from `c_dft_factory.py` and `d_trainer.py` is moved into functions within `tasks.py`, which are decorated with `@celery.task`. For example, `run_dft(dft_input_dict)` will be a Celery task. This decouples the task definition from the execution logic.
    *   To run the system, users will now need to start one or more Celery workers in the background (e.g., `celery -A mlip_autopipec.celery_app worker -l info`). These workers will automatically connect to the Redis broker and wait for jobs.
2.  **Advanced Goal Execution**:
    *   The user specifies a goal like `'diffusion'` in their `input.yaml`.
    *   The **Heuristic Engine** now has a much more sophisticated rule set. It recognizes the `'diffusion'` goal and generates a `SystemConfig` that specifies a multi-step workflow.
    *   The `OTFRunner` (Module E) is refactored into a more general `WorkflowManager`. Instead of a single, monolithic `run()` method, it will have methods corresponding to each goal (e.g., `run_diffusion_workflow()`).
    *   **Example Diffusion Workflow**:
        a.  The `run_diffusion_workflow` starts. It first runs a series of short MD simulations at different temperatures (e.g., 800K, 1000K, 1200K) to equilibrate the system. The OTF active learning loop is active during this phase to ensure the potential is reliable at these temperatures.
        b.  Once the system is equilibrated and the potential is stable, the workflow manager launches a long NVT production run for each temperature.
        c.  During these long runs, it saves the atomic trajectories to a file.
        d.  After the simulations are complete, the manager calls the analysis functions in `analysis/diffusion.py`.
        e.  The `diffusion.py` module reads the trajectory files, calculates the Mean Squared Displacement (MSD) of the atoms over time, and fits the resulting curve to determine the diffusion coefficient `D`.
        f.  The final result (`D` vs. `T`) is saved to a results file.
3.  **Data Flow**:
    *   The `WorkflowManager` (formerly `OTFRunner`) no longer starts its own worker processes. When it needs a DFT calculation, it calls `.delay()` or `.apply_async()` on the Celery task (e.g., `tasks.run_dft.delay(dft_input_dict)`). This places the task onto the Redis queue.
    *   A Celery worker, running in a separate process, picks up the task from the queue, executes the DFT calculation, and stores the result in the database.
    *   The main workflow can either poll the results backend for completion or move on to other tasks. This makes the entire system more scalable and resilient.

This new architecture provides a clear framework for adding new, complex, multi-step scientific workflows and ensures the system can scale to handle the massive number of computations these workflows require.

## 3. Design Architecture

The design for CYCLE05 focuses on creating a flexible workflow system and robust, well-tested analysis modules.

**Celery Integration Design:**

*   `celery_app.py`: Will contain the Celery app instance: `celery = Celery('mlip_autopipec', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')`. It will also configure task serialization (e.g., to use `json` or `pickle`).
*   `tasks.py`:
    *   The functions will be defined to be pure and stateless. They will take simple, serializable Python objects as input (e.g., dictionaries instead of Pydantic models, as Celery serialization can be tricky with complex objects).
    *   `run_dft(dft_input_dict)`: Takes a dictionary representing a `DFTInput`, runs the calculation, and returns a dictionary representing the `DFTOutput`. The task itself will be responsible for saving the result to the database.
    *   `run_training(trainer_config_dict)`: Takes a trainer configuration, pulls all necessary data from the database, runs the training, and saves the resulting potential.
    *   **Error Handling**: Celery's built-in retry mechanisms will be used. Tasks will be configured to automatically retry on specific, transient errors (e.g., a temporary network issue when accessing a file).

**Workflow Manager Design (in `e_inference.py`):**

*   The `OTFRunner` class is renamed to `WorkflowManager`.
*   The `run()` method becomes a dispatcher, calling other methods based on the `simulation_goal` from the `SystemConfig`.
*   `_run_single_md()`: A private helper method that encapsulates the core OTF loop for a single MD run. This is the logic from CYCLE03. It will now be called by the more complex workflow methods.
*   `run_elastic_workflow()`:
    1.  First, it calls `_run_single_md()` to get a fully relaxed, equilibrium structure at 0K.
    2.  It then uses the `ase.constraints.Deformation` class or similar logic in `analysis/elastic.py` to generate a list of specifically strained structures (e.g., shear strains, uniaxial strains) needed to compute the elastic tensor.
    3.  It dispatches a series of DFT calculation jobs for these structures using Celery.
    4.  Once all calculations are complete, it retrieves the stress tensors and passes them to a function in `analysis/elastic.py` that fits them to calculate the `Cij` matrix.
    5.  The final elastic tensor is saved.

**Analysis Modules Design (`analysis/`):**

*   Each module will contain pure functions that take data (e.g., file paths, NumPy arrays) as input and return calculated properties. They will have no knowledge of the workflow or the task queue.
*   `diffusion.py`:
    *   `calculate_msd(trajectory_file)`: Reads an ASE trajectory, calculates the MSD for each atom type, and returns the results as NumPy arrays.
    *   `fit_diffusion_coefficient(msd_data, temperature, time_step)`: Performs a linear fit on the MSD-vs-time data to get the diffusion coefficient.
*   `elastic.py`:
    *   `generate_strained_structures(equilibrium_structure)`: Creates the list of required deformations.
    *   `fit_elastic_constants(strains, stresses)`: Takes the list of applied strains and measured DFT stresses and solves the linear system to get the elastic constants.
*   This design makes the analysis logic highly reusable and easy to unit-test independently of the main application.

## 4. Implementation Approach

The implementation will be tackled by first swapping the queueing system, as this is a fundamental architectural change, and then implementing the new workflows and analysis modules on top of this new foundation.

**Step 1: Celery and Redis Integration**
*   Install Celery and Redis (`pip install celery redis`).
*   Set up a Redis server (e.g., via Docker for local development).
*   Implement `celery_app.py` and `tasks.py`. Port the core logic from the old `DFTWorker` and `TrainerWorker` into the new Celery tasks. Ensure the inputs and outputs are serializable.
*   Refactor the `WorkflowManager` (`e_inference.py`) to replace its `multiprocessing` calls with Celery task calls (e.g., `run_dft.delay(...)`).
*   Test this integration thoroughly. Run a simple equilibration workflow from CYCLE04 and verify that the DFT jobs are correctly placed on the Redis queue, executed by a Celery worker, and that the results are correct. This is a critical step to ensure the new foundation is solid.

**Step 2: Implement Analysis Modules**
*   Create the `analysis/` directory.
*   Develop the `diffusion.py` module. Write the `calculate_msd` and `fit_diffusion_coefficient` functions. Create unit tests for these functions in `tests/test_analysis/test_diffusion.py` using pre-computed, sample trajectory data.
*   Develop the `elastic.py` module and its associated unit tests in a similar manner.

**Step 3: Implement Advanced Workflows**
*   Expand the `HeuristicEngine` to recognize the new goals (`'diffusion'`, `'elastic'`) and to generate the appropriate, detailed `SystemConfig` for them.
*   In the `WorkflowManager`, implement the new `run_diffusion_workflow` and `run_elastic_workflow` methods.
*   These methods will orchestrate the entire process: calling the OTF loop for equilibration, running production MDs, dispatching analysis tasks, and saving the final results.

**Step 4: Update the UI**
*   Update the CLI in `cli.py` to accept the new simulation goals. Add validation to ensure only supported goals are accepted.
*   Update the Dashboard. The dashboard can be improved to show the status of the Celery queues (e.g., number of pending vs. completed tasks). The `flower` tool, which is a standard part of the Celery ecosystem, can also be recommended to users for more detailed monitoring. New tabs or sections can be added to the dashboard to display the final results of the new analysis types (e.g., a plot of the diffusion coefficient vs. temperature).

## 5. Test Strategy

Testing in CYCLE05 needs to cover both the new, complex scientific workflows and the robustness of the new distributed tasking system.

**Unit Testing Approach (Min 300 words):**

Unit tests will focus on the pure, algorithmic components of the new analysis modules.

*   **Analysis Modules (`test_analysis/`):**
    *   **Diffusion:** In `test_diffusion.py`, we will create a fake `ase.io.Trajectory` file programmatically. This trajectory will represent ideal Brownian motion, where the mean squared displacement increases linearly with time, i.e., `<r^2> = 6Dt`. We will write a test that generates this artificial trajectory for a known diffusion coefficient `D`. We will then pass this file to our `calculate_msd` function and assert that the output MSD is a straight line. We will then pass this MSD data to our `fit_diffusion_coefficient` function and assert that the fitted `D` is very close to the original `D` used to generate the data. This provides a rigorous, quantitative validation of the entire analysis chain.
    *   **Elastic:** In `test_elastic.py`, we will test the fitting logic. We will take a known elastic tensor for a cubic material (like Aluminum), which has only three independent constants ($C_{11}, C_{12}, C_{44}$). We will generate a set of strains and then calculate the "perfect" corresponding stresses using the known elastic tensor. We will then pass these strains and stresses to our `fit_elastic_constants` function and assert that it recovers the original $C_{11}, C_{12}, C_{44}$ values to high precision. This validates the correctness of the fitting algorithm.

*   **Heuristic Engine:** We will add new unit tests to `test_heuristic_engine.py` to validate the logic for the new simulation goals, similar to the tests in CYCLE04. For a `'diffusion'` goal, we will assert that the generated config specifies multiple, high-temperature MD runs.

**Integration Testing Approach (Min 300 words):**

Integration tests will verify the full workflows and the correct functioning of the Celery-based task management.

*   **Test Scenario 1: Celery Infrastructure Test**
    *   This is a technical test to ensure the new queueing system works. We will run the simple "equilibrate" workflow from CYCLE04.
    *   The test will start a temporary Redis server and a Celery worker in the background.
    *   It will then launch the main workflow.
    *   Using Celery's inspection API, the test will assert that DFT tasks are being placed on the queue and that their state changes from "PENDING" to "SUCCESS".
    *   It will check the final result (the equilibrium lattice constant) to ensure that the outcome is the same as it was with the old `multiprocessing` system, proving the new infrastructure works correctly.

*   **Test Scenario 2: End-to-End Elastic Constants Workflow**
    *   **Objective**: To run the complete workflow for calculating the elastic constants of a known material, like Silicon, and verify the result.
    *   **Execution**:
        1.  The test will start with a minimal `input.yaml` specifying Silicon and the goal `'elastic'`.
        2.  It will launch the `mlip-auto run` command.
        3.  The `WorkflowManager` will first perform an equilibration run to get the relaxed structure (this part will use the OTF loop and a mock DFT calculator).
        4.  It will then generate the set of ~6-12 required strained structures.
        5.  It will dispatch these as DFT jobs to the Celery queue.
        6.  The test will wait for all Celery jobs to complete.
        7.  The manager will then collect the results and call the `fit_elastic_constants` analysis function.
    *   **Validation**:
        1.  The test will assert that the correct number of DFT jobs were created and completed successfully in the Celery queue.
        2.  The primary validation will be on the final result. The test will parse the output file containing the calculated elastic tensor for Silicon and assert that the values for $C_{11}, C_{12}, C_{44}$ are within a small tolerance (e.g., 5%) of their well-established experimental/DFT literature values. This provides a full, end-to-end validation of this complex new scientific workflow.
