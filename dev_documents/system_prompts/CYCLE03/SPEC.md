# CYCLE03: The Training Engine (SPEC.md)

## 1. Summary

This document outlines the detailed technical specifications for Cycle 3 of the MLIP-AutoPipe project. The focus of this cycle is to develop the **Training Engine**, the component responsible for consuming the DFT data and producing a trained Machine Learning Interatomic Potential (MLIP). This represents the heart of the "learning" part of the active learning loop. While previous cycles focused on generating and curating high-quality data, this cycle puts that data to use. It bridges the gap between raw quantum mechanical results and a fast, accurate, and usable surrogate model for large-scale simulations.

The primary deliverable for this cycle is a `PacemakerTrainer` class. This class will encapsulate all the logic required to interact with the external Pacemaker training code. Its main responsibility will be to orchestrate the training workflow, which involves several key steps. First, it must read the accumulated DFT data from the central ASE database, which was populated in Cycle 1. Second, it must dynamically generate the necessary input configuration files for the Pacemaker executable. This configuration includes critical training hyperparameters such as the relative weights of energy, forces, and stress in the loss function, and settings for advanced features like Delta Learning.

Third, the `PacemakerTrainer` will be responsible for invoking the Pacemaker training process as a secure and monitored subprocess. It will capture the output logs to check for successful completion or to report errors. Finally, upon successful training, it will locate the generated potential file (typically with a `.yace` extension) and make it available for the next stages of the workflow.

This cycle is a major milestone because it creates the first complete, albeit "open-loop," pipeline. By the end of this cycle, the system will be able to go from a set of atomic structures all the way to a trained potential, ready for use. This sets the stage for Cycle 4, where this potential will be deployed in a live simulation to close the active learning loop.

## 2. System Architecture

The architecture for Cycle 3 introduces a new, specialised module for training, `mlip_autopipec/modules/training.py`. This module will depend on the database utilities created in Cycle 1 and will be orchestrated by the main workflow manager in a future cycle.

**File Structure for Cycle 3:**

The following ASCII tree highlights the new files to be created in this cycle in **bold**.

```
mlip-autopipe/
├── dev_documents/
│   └── system_prompts/
│       ├── CYCLE01/
│       ├── CYCLE02/
│       └── CYCLE03/
│           ├── **SPEC.md**
│           └── **UAT.md**
├── mlip_autopipec/
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── dft.py
│   │   ├── exploration.py
│   │   └── **training.py**     # Core PacemakerTrainer implementation
│   └── utils/
│       └── ase_utils.py
├── tests/
│   └── modules/
│       ├── __init__.py
│       ├── test_dft.py
│       ├── test_exploration.py
│       └── **test_training.py**  # Unit and integration tests for training.py
└── pyproject.toml
```

**Component Blueprint: `modules/training.py`**

This file will contain the `PacemakerTrainer` class.

-   **`PacemakerTrainer` class:**
    -   **`__init__(self, training_config)`**: The constructor takes a `TrainingConfig` Pydantic model. This object will specify all necessary parameters, such as the path to the Pacemaker executable, the database path, and all training hyperparameters. This dependency injection approach keeps the class decoupled and highly testable.
    -   **`train(self) -> Path`**: This is the primary public method. It executes the entire training workflow and, upon success, returns the `Path` object pointing to the newly created potential file (`.yace`).
    -   **`_read_data_from_db(self) -> List[ase.Atoms]`**: A private helper method that connects to the ASE database specified in the configuration and reads all stored atomic structures into a list. It will use the utilities defined in `ase_utils.py`.
    -   **`_prepare_pacemaker_input(self, training_data: List[ase.Atoms], working_dir: Path) -> None`**: This method is responsible for setting up the training directory. It will:
        1.  Write the `training_data` to a `.xyz` file in the format expected by Pacemaker.
        2.  Generate the `pacemaker.in` configuration file. This will be done using a template engine (like Jinja2) to combine a base template with the specific hyperparameters from the `TrainingConfig` model.
    -   **`_execute_training(self, working_dir: Path) -> subprocess.CompletedProcess`**: This method invokes the external `pacemaker` binary. It will change the current working directory to `working_dir` for the duration of the subprocess call. It uses `subprocess.run` to execute the command, capturing `stdout` and `stderr` for logging and diagnostics. It will raise an exception if the training code returns a non-zero exit code.
    -   **`_find_output_potential(self, working_dir: Path) -> Path`**: After a successful training run, this method scans the `working_dir` to find the output potential file. It will typically look for a file ending in `.yace` and return its absolute path. If no such file is found, it will raise an error, as this indicates the training process failed silently.

This modular design ensures a clear separation of concerns: reading data, preparing inputs, executing the external process, and handling the output are all distinct, testable steps.

## 3. Design Architecture

The design of the Training Engine is guided by the project's Schema-First philosophy. The entire training process is configured via a strict Pydantic model, ensuring that all hyperparameters are valid and well-defined before the potentially time-consuming training process is launched.

**Pydantic Schema Definitions:**

The following Pydantic models will be added to `mlip_autopipec/config/models.py`.

1.  **`LossWeights(BaseModel)`**: A nested model to define the relative weights in the loss function.
    -   `energy: float = Field(1.0, gt=0)`: Weight for the energy term.
    -   `forces: float = Field(100.0, gt=0)`: Weight for the force components.
    -   `stress: float = Field(10.0, gt=0)`: Weight for the stress tensor components.

2.  **`TrainingConfig(BaseModel)`**: The main configuration object for the `PacemakerTrainer`.
    -   `pacemaker_executable: FilePath`: The path to the Pacemaker training binary. `FilePath` ensures the executable exists.
    -   `data_source_db: Path`: The path to the ASE database containing the training data.
    -   `template_file: FilePath`: The path to the Jinja2 template for the `pacemaker.in` file.
    -   `delta_learning: bool = True`: A flag to enable or disable learning relative to a ZBL baseline.
    -   `loss_weights: LossWeights = Field(default_factory=LossWeights)`: The nested model for loss function weights.
    -   `model_config = ConfigDict(extra='forbid')`: Enforces strictness in the configuration.

**Data Flow and Consumers:**

-   **Producer of Data:** The ASE Database, populated by the `DFTFactory` (Cycle 1) and curated by the `SurrogateExplorer` (Cycle 2), is the source of training data.
-   **Consumer of Data:** The `PacemakerTrainer` is the primary consumer of the data in the ASE database.
-   **Producer of Model:** The `PacemakerTrainer` produces the final MLIP file (`.yace`).
-   **Consumer of Model:** The `LammpsRunner` (to be implemented in Cycle 4) will be the primary consumer of the generated `.yace` file, using it to run molecular dynamics simulations.
-   **Trigger:** The `WorkflowManager` (future cycle) will be responsible for instantiating and calling the `PacemakerTrainer.train()` method whenever sufficient new data has been added to the database.

**Invariants and Constraints:**
-   A training run will not be attempted if the number of structures in the database is below a certain threshold (e.g., 50). This prevents wasted effort on training a model with insufficient data.
-   The `template_file` must contain the correct Jinja2 placeholders corresponding to the fields in the `TrainingConfig` and `LossWeights` models.
-   All structures read from the database for training must contain forces. Structures without forces will be ignored or will cause an error.

This design ensures that the training process is reproducible, configurable, and robust, with clear data contracts between the different parts of the overall system.

## 4. Implementation Approach

The implementation will proceed logically, starting with configuration, then data handling, and finally process execution.

1.  **Add Dependencies:** If a template engine like Jinja2 is to be used, it will be added to the project's dependencies in `pyproject.toml`.

2.  **Update Pydantic Models:** The first step is to add the `LossWeights` and `TrainingConfig` models to the `mlip_autopipec/config/models.py` file.

3.  **Create `pacemaker.in` Template:** A template file (`pacemaker.in.j2`) will be created. This file will look like a standard `pacemaker.in` file, but with Jinja2 placeholders for values that will be configured at runtime.
    ```ini
    # pacemaker.in.j2
    ...
    delta_learn = {{ config.delta_learning | lower }}
    ...
    [loss_weights]
    energy = {{ config.loss_weights.energy }}
    forces = {{ config.loss_weights.forces }}
    stress = {{ config.loss_weights.stress }}
    ...
    ```

4.  **Implement Data Reading:** In `modules/training.py`, implement the `_read_data_from_db` helper method. This will use `ase.db.connect(self.config.data_source_db)` and a simple loop to read all rows into a list of `ase.Atoms` objects.

5.  **Implement Input Preparation:** Implement the `_prepare_pacemaker_input` method.
    -   It will first use ASE's `ase.io.write` to save the list of `ase.Atoms` objects to a file named `training_data.xyz` inside the specified `working_dir`.
    -   It will then load the Jinja2 template, render it with the `TrainingConfig` object, and write the result to `pacemaker.in` inside the `working_dir`.

6.  **Implement Training Execution:** Implement the `_execute_training` method. It will use `subprocess.run`, setting the `cwd` (current working directory) argument to the `working_dir`. It will also set `check=True` to ensure that an exception is raised automatically if the subprocess returns a non-zero exit code.

7.  **Implement Output Handling:** Implement the `_find_output_potential` method. It will use `pathlib.Path.glob` to search for `*.yace` files within the `working_dir`. It should handle cases where zero or more than one potential file is found.

8.  **Assemble the `PacemakerTrainer` Class:** Finally, implement the main `train()` method. This method will create a temporary directory for the training run, then call the helper methods in sequence: `_read_data_from_db`, `_prepare_pacemaker_input`, `_execute_training`, and `_find_output_potential`. It will return the path provided by the final method.

This structured approach ensures that each part of the complex training orchestration is handled by a dedicated, single-responsibility function, which is easier to debug and test.

## 5. Test Strategy

Testing the `PacemakerTrainer` involves a combination of unit tests with extensive mocking to validate the logic, and a targeted integration test to verify the interaction with the external training code.

**Unit Testing Approach (Min 300 words):**

Unit tests in `tests/modules/test_training.py` will focus on the orchestration logic of the `PacemakerTrainer` without actually running the expensive training process.

-   **Testing Input File Generation:** The most critical unit test will be for the `_prepare_pacemaker_input` method. We will create a test that instantiates a `PacemakerTrainer` with a specific `TrainingConfig` object. It will then call the preparation method, pointing it to a temporary directory. The test will then read the generated `pacemaker.in` file from this directory and assert that its contents are exactly as expected. For example, it will assert that the line `energy = 1.0` is present if the `LossWeights` were set to their default values. It will also check that the `training_data.xyz` file was created.

-   **Mocking Process Execution:** The main `train()` method will be tested by mocking `subprocess.run`.
    -   A test for the "happy path" (`test_train_success`) will configure the mock to return a `CompletedProcess` with a return code of 0. The test will then assert that the `train` method completes without raising an exception and returns a path. We can also mock `_find_output_potential` to return a dummy path to ensure the full chain is tested.
    -   A test for the failure path (`test_train_failure`) will configure the mock to raise a `CalledProcessError` (or to have a non-zero return code). The test will then use `pytest.raises` to assert that the `train` method correctly raises a custom exception (e.g., `TrainingError`).

-   **Mocking Database Interaction:** We can test the `_read_data_from_db` method by creating a mock ASE database object and patching `ase.db.connect`. This allows us to test the data reading logic without needing a real database file on disk.

**Integration Testing Approach (Min 300 words):**

The integration test provides the ultimate confidence that the trainer can successfully interact with the real Pacemaker executable. While potentially slow, a single end-to-end test is invaluable.

-   **End-to-End Mini-Training Run:** This test, marked with `@pytest.mark.integration`, will validate the entire process on a small scale.
    1.  **Setup:** The test will first create a temporary directory. Inside it, it will create a "toy" ASE database and populate it with a small number (e.g., 20-30) of pre-calculated, valid `ase.Atoms` objects for a simple system like Silicon. This provides a self-contained, reproducible dataset.
    2.  **Configuration:** A `TrainingConfig` object will be created, pointing to the real `pacemaker` executable and the toy database created in the setup step. The training parameters in the template will be set for a very short, fast run (e.g., only a few iterations).
    3.  **Execution:** The test will instantiate a `PacemakerTrainer` with this configuration and call the `train()` method.
    4.  **Assertion:** The assertions will verify the successful completion of the process. The test will assert that the `train` method does not raise an exception. Most importantly, it will assert that the returned path exists, is a file, and has the `.yace` extension. A more advanced assertion could even use ASE to try and load the resulting potential file to ensure it is not corrupted. This test provides a definitive "yes/no" answer to the question: "Can our trainer successfully produce a potential?"
