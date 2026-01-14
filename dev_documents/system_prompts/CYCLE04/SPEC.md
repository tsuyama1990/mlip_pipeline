# MLIP-AutoPipe: Cycle 04 Specification

- **Cycle**: 04
- **Title**: The Factory - Pacemaker Trainer and Database Loop
- **Status**: Scoping

---

## 1. Summary

Cycle 04 marks a pivotal moment in the project: the creation of the first Machine Learning Interatomic Potential. The preceding cycles have established a robust pipeline for generating and selecting candidate structures, and for calculating their properties using a DFT engine. Now, we must consume that valuable data and use it to train a model. This cycle focuses on implementing **Module D: the Pacemaker Trainer**, the component responsible for orchestrating the MLIP training process. The primary objective is to create a seamless, automated workflow that queries the project's database for all completed DFT calculations, configures a training job based on the system's parameters, executes the training using the Pacemaker framework, and manages the resulting potential file.

The implementation will be centered around a new `PacemakerTrainer` class. This class will act as a bridge between our internal data and configuration formats and the external `pacemaker_train` command-line tool. Its core responsibilities are threefold. First, it must communicate with the `DatabaseManager` (from Cycle 01) to fetch all the atomic structures that have been successfully processed by the DFT Factory. Second, it will be responsible for dynamically generating the YAML configuration file required by Pacemaker. This is a critical step, as it translates the high-level parameters and design choices from our `SystemConfig` (e.g., loss function weights, radial basis function parameters) into the specific format that the Pacemaker executable understands.

Third, the `PacemakerTrainer` will manage the execution of the training process itself, wrapping the `pacemaker_train` command in a subprocess call. It will monitor the process for successful completion and identify the path to the newly created potential file (a `.yace` file). By the end of this cycle, we will have closed the initial data-to-model loop. The system will be able to take the diverse structures selected in Cycle 03, submit them to the (mocked) DFT factory, and then use the results to train and produce a tangible, version-one MLIP. This forms the foundation for the full active learning loop to be implemented in subsequent cycles.

---

## 2. System Architecture

This cycle introduces the training module, which directly interacts with the database component from Cycle 01 and consumes the data produced by the pipeline so far.

**File Structure for Cycle 04:**

The following files will be created or modified. New files are marked in **bold**.

```
.
└── src/
    └── mlip_autopipec/
        ├── config/
        │   └── system.py       # Modified to add TrainerParams
        ├── data/
        │   └── database.py
        └── modules/
            ├── explorer.py
            └── **trainer.py**      # Module D: PacemakerTrainer class
```

**Component Breakdown:**

*   **`config/system.py`**: The `SystemConfig` Pydantic model will be extended with a `TrainerParams` sub-model. This schema will define all parameters related to the training process. This includes loss function weights (for energy, forces, and stress), hyperparameters for the ACE potential (e.g., radial basis cutoff, correlation order), and settings for the training optimiser.

*   **`modules/trainer.py`**: This new file will contain the `PacemakerTrainer` class. It will be initialized with the `SystemConfig` and the `DatabaseManager`. Its primary public method, `train()`, will encapsulate the entire training workflow. It will not take any arguments, as it is designed to pull all necessary information from the database and the configuration object it was initialized with. The method will return the file path to the newly generated `.yace` potential file upon successful completion.

This architecture clearly isolates the training logic. The trainer module does not need to know how the DFT data was generated or selected; it only needs to know how to retrieve it from the database and how to configure the Pacemaker executable.

---

## 3. Design Architecture

The `PacemakerTrainer` is designed as a stateless orchestrator. It coordinates the actions of retrieving data, writing a configuration file, and running an external process.

**Pydantic Schema Design (`system.py` extension):**

*   **`TrainerParams`**: This new `BaseModel` in `SystemConfig` will be the definitive source for all training settings.
    *   **`LossWeights`**: A nested model to specify the relative weights, e.g., `energy: float = 1.0`, `forces: float = 100.0`, `stress: float = 10.0`. These defaults reflect the higher importance of accurate forces.
    *   **`ACEParams`**: A nested model defining the potential's architecture, such as `radial_basis`, `correlation_order`, and `element_dependent_cutoffs`.
    *   **Producers and Consumers**: The `HeuristicEngine` produces these parameters based on the user's high-level goal, and the `PacemakerTrainer` is the sole consumer.

**`PacemakerTrainer` Class Design (`trainer.py`):**

*   **Interface**: `__init__(self, config: SystemConfig, db_manager: DatabaseManager)` and `train() -> str`. The constructor uses dependency injection, making the class easier to test by allowing a mock `DatabaseManager` to be passed in. The `train` method's string return value is the explicit path to the final artefact.
*   **Internal Logic**: The `train` method is a sequence of non-trivial, coordinated steps:
    1.  **`_fetch_training_data`**: A private method that calls the `db_manager.get_completed_calculations()` to retrieve a list of all `Atoms` objects. It will then save this list to a temporary `.xyz` file, as this is the format Pacemaker consumes. This method returns the path to this temporary file.
    2.  **`_generate_pacemaker_config`**: This method is responsible for the translation logic. It reads from `self.config.trainer` and programmatically builds the dictionary required for the Pacemaker YAML file. It will then use a YAML library to dump this dictionary to a temporary file (e.g., `pacemaker_input.yaml`). This ensures the output is always a valid YAML file. The method returns the path to this config file.
    3.  **`_execute_training`**: This method takes the paths to the data file and the config file as arguments. It will construct the command-line call, for example, `['pacemaker_train', '--config-file', 'path/to/config.yaml']`. It will then use `subprocess.run` to execute this command, capturing `stdout` and `stderr` and checking the return code. If the process fails, it will raise a `TrainingFailedError` with the captured output for debugging. Upon success, it will parse the `stdout` to find the path of the generated `.yace` file, which Pacemaker prints upon completion.
*   **File Management**: The class will be responsible for managing the temporary files it creates (the `.xyz` data file and the `.yaml` config file), ensuring they are cleaned up after the training process is complete, regardless of success or failure. This can be achieved using a `try...finally` block.

This design makes the training process a robust, transactional operation. It's either completed successfully, yielding a potential file, or it fails with a clear error, leaving no mess behind.

---

## 4. Implementation Approach

The implementation will focus on the three core responsibilities of the `PacemakerTrainer`: data fetching, config generation, and process execution.

1.  **Dependencies and Configuration:**
    *   Add any necessary YAML library (like `pyyaml`) to the `pyproject.toml`. The `pacemaker` library itself will also be a dependency.
    *   Implement the `TrainerParams` and its nested Pydantic models in `src/mlip_autopipec/config/system.py`.

2.  **Scaffold the Trainer Class (`trainer.py`):**
    *   Create the new file `src/mlip_autopipec/modules/trainer.py`.
    *   Define the `PacemakerTrainer` class. Its `__init__` will store the `config` and `db_manager`.
    *   Define the empty private methods: `_fetch_training_data`, `_generate_pacemaker_config`, and `_execute_training`.

3.  **Implement Data Fetching:**
    *   In `_fetch_training_data`, call the `db_manager` to get the list of `Atoms` objects.
    *   Use ASE's `ase.io.write` function to save this list to a named temporary file. Use Python's `tempfile` module to handle the file creation securely.

4.  **Implement Config Generation:**
    *   In `_generate_pacemaker_config`, create a Python dictionary that mirrors the structure of the `pacemaker_train.yaml` file.
    *   Populate this dictionary with values from `self.config.trainer`. For example: `{'loss_weights': {'forces': self.config.trainer.loss_weights.forces}}`.
    *   Crucially, add the path to the temporary data file (created in the previous step) to this dictionary.
    *   Use the `yaml.dump` function to write this dictionary to another named temporary file.

5.  **Implement Training Execution:**
    *   In `_execute_training`, build the command-line arguments list.
    *   Use `subprocess.run` with `check=True`, `capture_output=True`, and `text=True` to execute the command. This ensures that an exception is raised automatically on failure and that the output is decoded as a string.
    *   Use a regular expression to parse the captured `stdout` to find the line indicating the output file path.

6.  **Orchestrate and Finalise:**
    *   In the main `train` method, implement the `try...finally` block for cleanup.
    *   Inside the `try` block, call the private methods in order: fetch data, generate config, execute training.
    *   Return the path of the potential file.
    *   The `finally` block will contain the logic to delete the temporary files.

7.  **Write Tests (`tests/modules/test_trainer.py`):**
    *   Create the new test file.
    *   Write a unit test for `_generate_pacemaker_config`. Provide a sample `SystemConfig` and assert that the generated YAML string is exactly correct.
    *   Write an integration test for the `train` method, mocking the `DatabaseManager` and the `subprocess.run` call.

---

## 5. Test Strategy

Testing the `PacemakerTrainer` involves verifying its ability to correctly interface with the database and the external training process.

**Unit Testing Approach (Min 300 words):**

The primary unit test will focus on the configuration generation logic, which is the most complex piece of pure-Python code in this module.

*   **Trainer Configuration Generation (`tests/modules/test_trainer.py`):**
    The test function `test_generate_pacemaker_config` will ensure the translation from our `SystemConfig` to the Pacemaker YAML format is flawless. We will create a `pytest` fixture that provides a detailed `SystemConfig` object with specific, non-default values for trainer parameters (e.g., custom loss weights, a high correlation order for ACE). The test will instantiate the `PacemakerTrainer` with this config. It will then call the private `_generate_pacemaker_config` method. The core of the test will be to parse the resulting YAML file (using `yaml.safe_load`) back into a Python dictionary. We will then assert that the values in this dictionary are correct. For example, we'll assert that `parsed_yaml['fit_params']['loss_weights']['forces']` is equal to the value we set in our test's `SystemConfig`. We will also assert that the `dataset_filename` key in the YAML correctly points to the temporary data file path that the method should have generated. This test guarantees that our trainer can correctly configure the external tool.

**Integration Testing Approach (Min 300 words):**

The integration test will verify the end-to-end orchestration logic of the `train` method, from data query to process execution, using mocks to isolate the trainer's behaviour.

*   **End-to-End Training Orchestration (`tests/modules/test_trainer.py`):**
    The test `test_train_orchestration` will verify the entire workflow.
    1.  **Mock `DatabaseManager`**: We will create a mock `DatabaseManager` object using `mocker.Mock`. We will configure its `get_completed_calculations` method to return a simple list containing a few dummy ASE `Atoms` objects.
    2.  **Mock `subprocess.run`**: We will patch `subprocess.run`. The mock will be configured to simulate a successful run (`returncode=0`). Its `stdout` will be set to a multi-line string that mimics the real output of `pacemaker_train`, including a line like `INFO: Final potential saved to: /path/to/test_potential.yace`.
    3.  **Execution**: We will instantiate `PacemakerTrainer` with a test `SystemConfig` and our mock `DatabaseManager`. We will then call the public `train()` method.
    4.  **Assertions**: We will make several critical assertions. First, we'll assert that the mock `db_manager.get_completed_calculations` was called exactly once. Second, we'll assert that the mocked `subprocess.run` was called with the correct command-line arguments, including the path to the auto-generated config file. Finally, we will assert that the return value of the `train()` method is the string `'/path/to/test_potential.yace'`, proving that the output parsing worked correctly. We will also have a separate test for the failure case, where the mock is configured with a non-zero return code, and we assert that our custom `TrainingFailedError` is raised.
