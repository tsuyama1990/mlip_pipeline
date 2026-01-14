# MLIP-AutoPipe: Cycle 04 User Acceptance Testing

- **Cycle**: 04
- **Title**: The Factory - Pacemaker Trainer and Database Loop
- **Status**: Design

---

## 1. Test Scenarios

User Acceptance Testing for Cycle 04 is designed to deliver a moment of profound satisfaction: the creation of the very first Machine Learning Interatomic Potential. This UAT provides the first tangible, valuable artefact from the entire pipeline. The user, acting as a developer or scientist, will witness the culmination of the previous cycles' work: data is automatically retrieved from the database, a training configuration is generated on the fly, and a training process is launched to produce a usable MLIP. This process is designed to amaze the user by making the complex, often manual, process of training a potential look simple, robust, and completely automated. The UAT will be conducted in a Jupyter Notebook (`04_first_potential_training.ipynb`), which will guide the user through triggering the training and then immediately using the resulting potential to make a prediction.

---

### **Scenario ID: UAT-C04-01**
- **Title**: Training and Verifying the First MLIP
- **Priority**: Critical

**Description:**
This scenario guides the user through the automated training process and provides immediate validation that the resulting potential is functional. The user will start by inspecting a small, pre-populated database of DFT-calculated structures. They will then instantiate and run the `PacemakerTrainer`, which will automatically find the data, configure the training, and produce a `.yace` file. The climax of the UAT is the final step, where the notebook loads the newly created potential back into memory, applies it to an `Atoms` object, and successfully calculates an energy. This provides a direct, satisfying confirmation that the entire training pipeline is working end-to-end.

**UAT Steps via Jupyter Notebook (`04_first_potential_training.ipynb`):**

**Part 1: The Training Dataset**
*   The notebook will begin by importing `SystemConfig`, `DatabaseManager`, `PacemakerTrainer`, and `ase.io`.
*   **Step 1.1:** The UAT will start with a pre-prepared SQLite database file (`training_data.db`) that contains a few (e.g., 10-20) structures with pre-calculated energy and forces, simulating the output of the previous pipeline stages.
*   **Step 1.2:** The notebook will instantiate the `DatabaseManager` and use it to connect to the database. It will then query the database and print the number of available structures, allowing the user to confirm that the training data is present.

**Part 2: Automated Training**
*   **Step 2.1:** The user will create a `SystemConfig` object that includes the necessary `TrainerParams` for the training run.
*   **Step 2.2:** The notebook will instantiate the `PacemakerTrainer`, providing it with the config and the database manager.
*   **Step 2.3:** In a single, powerful cell, the user will execute `potential_file_path = trainer.train()`. The notebook will explain that, for the UAT, the actual `pacemaker_train` command is mocked to run instantaneously and produce a pre-canned `.yace` file. This is to ensure the UAT is fast and reliable. The cell's output will print the path to the newly created potential file. The user is amazed by the simplicity of the process—one command to orchestrate the entire training job.

**Part 3: Verifying the Potential**
*   **Step 3.1 (The Payoff):** This is the most important step. The notebook will demonstrate how to use the newly created potential. It will import the necessary calculator tools from the `pacemaker` library.
*   **Step 3.2:** It will load the potential from the `potential_file_path` returned in the previous step.
*   **Step 3.3:** It will create a new, simple `Atoms` object (e.g., a 2-atom Ni dimer).
*   **Step 3.4:** It will attach the loaded potential to this `Atoms` object as its calculator.
*   **Step 3.5:** The final cell will call `atoms.get_potential_energy()`. The successful return of a floating-point number is the ultimate verification. It proves, in a single, undeniable step, that the trainer produced a valid, loadable, and executable Machine Learning Interatomic Potential. The UAT will conclude by stating that this potential, while trained on very little data, is the seed that the active learning loop will grow in the next cycle.

---

## 2. Behavior Definitions

These Gherkin-style definitions specify the expected behaviour of the `PacemakerTrainer`.

**Feature: Automated MLIP Training**
As a developer, I want to automatically train a Pacemaker potential using all available DFT data from the database, so that I can create a functional MLIP without manual configuration.

---

**Scenario: Successful Training from Database**

*   **GIVEN** a database containing 20 successfully completed DFT calculations.
*   **AND** a `SystemConfig` specifying valid training parameters.
*   **AND** a `DatabaseManager` connected to this database.
*   **AND** the `subprocess.run` call for `pacemaker_train` is mocked to simulate a successful execution.
*   **AND** the mock is configured to report that the final potential was saved to `results/test.yace`.
*   **WHEN** I instantiate the `PacemakerTrainer` and call its `train()` method.
*   **THEN** the `DatabaseManager`'s method to retrieve all calculations must be called.
*   **AND** a temporary XYZ file containing the 20 structures must be created.
*   **AND** a valid Pacemaker YAML config file must be generated, pointing to the temporary XYZ file.
*   **AND** the `pacemaker_train` command must be executed as a subprocess.
*   **AND** the `train()` method must return the string `results/test.yace`.

---

**Scenario: Handling Training Process Failure**

*   **GIVEN** a valid database and `SystemConfig`.
*   **AND** the `subprocess.run` call for `pacemaker_train` is mocked to simulate a *failed* execution (e.g., with a non-zero return code).
*   **AND** the mock is configured to have captured error messages in its `stderr`.
*   **WHEN** I call the `PacemakerTrainer`'s `train()` method.
*   **THEN** the system must raise a custom `TrainingFailedError` exception.
*   **AND** the exception's message must contain the `stderr` from the failed process to aid in debugging.
*   **AND** any temporary files (like the XYZ data file and YAML config) created before the failure must be deleted.

---

**Scenario: Handling No Data in Database**

*   **GIVEN** a database that contains zero completed DFT calculations.
*   **AND** a valid `SystemConfig`.
*   **WHEN** I call the `PacemakerTrainer`'s `train()` method.
*   **THEN** the system should not proceed with training.
*   **AND** it should return a special value, like `None`, or raise a specific `NoTrainingDataError`.
*   **AND** it must log a clear warning message stating that no training data was available.
