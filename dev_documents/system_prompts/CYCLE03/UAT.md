# CYCLE03: The Training Engine (UAT.md)

## 1. Test Scenarios

User Acceptance Testing (UAT) for Cycle 3 focuses on verifying that the Training Engine can successfully take a dataset of DFT calculations and produce a valid, usable Machine Learning Interatomic Potential (MLIP). The key user experience is one of confidence and verification: seeing the system transform raw data into a tangible, functional model. The UAT will be conducted within a Jupyter Notebook, which allows for clear, step-by-step execution and provides opportunities to inspect the inputs and outputs, thereby demystifying the training process.

| Scenario ID   | Test Scenario                                             | Priority |
|---------------|-----------------------------------------------------------|----------|
| UAT-C3-001    | **Successful End-to-End Training Run**                    | **High**     |
| UAT-C3-002    | **Verification of a Trained Potential**                   | **High**     |
| UAT-C3-003    | **Handling of Training Process Failure**                  | **Medium**   |

---

### **Scenario UAT-C3-001: Successful End-to-End Training Run**

**(Min 300 words)**

**Description:**
This is the primary "happy path" scenario for the training module. Its goal is to provide the user with a clear, observable demonstration of the entire training process from start to finish. The user will see the system gather data from a database, prepare the necessary files for the Pacemaker training code, execute the training, and finally report the location of the newly created potential file. This scenario is crucial for building trust in the automation, showing that the system can correctly orchestrate all the necessary steps without any manual intervention.

**UAT Steps in Jupyter Notebook:**
1.  **Setup:** The notebook will import the `PacemakerTrainer` class and other utilities. It will define paths for a temporary working directory and a test ASE database.
2.  **Prepare the Dataset:** The notebook will programmatically create a small but valid ASE database. It will add 20-30 `ase.Atoms` objects with pre-calculated energy and forces. This ensures the UAT is self-contained and reproducible. The notebook will display a message confirming the dataset's creation, for instance: "Created a test database with 25 Silicon structures."
3.  **Configure the Trainer:** An instance of the `TrainingConfig` Pydantic model will be created. The notebook will display this configuration object, making the settings for the run (like paths and learning flags) transparent to the user.
4.  **Instantiate and Run:** A `PacemakerTrainer` object will be instantiated with the configuration. The notebook will then call the `trainer.train()` method in a new cell.
5.  **Live Log Streaming (User Amazement):** To make the process engaging, the notebook will be configured to stream the `stdout` from the Pacemaker subprocess in real-time. The user will see the training log appear directly in the notebook output, showing the optimisation steps, the decreasing loss function (RMSE), and the final confirmation message from the training code. This provides a powerful visual confirmation that a complex process is running successfully.
6.  **Report the Outcome:** Upon successful completion, the `train` method will return the path to the potential file. The notebook will print a clear, congratulatory message: "✅ Training completed successfully! The new potential is located at: `/path/to/temporary/dir/potential.yace`". This tangible output is the key deliverable for the user.

---

### **Scenario UAT-C3-002: Verification of a Trained Potential**

**(Min 300 words)**

**Description:**
Simply creating a potential file is not enough; the user needs confidence that the potential is actually functional and has learned from the training data. This UAT scenario goes one step further than the previous one. After successfully training a potential, it will immediately load that potential and use it to predict the energy and forces for a structure from the original training set. By comparing the MLIP's prediction to the original DFT "ground truth," the user can directly verify that the model has learned correctly. This provides a powerful and intuitive measure of the model's quality and closes the loop on the training process.

**UAT Steps in Jupyter Notebook:**
1.  **Prerequisite:** This scenario runs immediately after UAT-C3-001 in the same notebook, using the `.yace` file that was just created.
2.  **Select a Test Structure:** The notebook will connect to the same test ASE database used for training and select one structure that was part of the training set. It will store the original DFT energy and forces from this structure as the "ground truth."
3.  **Load the New Potential:** The notebook will use a suitable library (like `pyace` or an ASE calculator interface for the `.yace` format) to load the potential from the file path returned by the trainer.
4.  **Perform Prediction:** An ASE calculator object will be created using the loaded MLIP. This calculator will be attached to the test structure, and the notebook will call `atoms.get_potential_energy()` and `atoms.get_forces()` to get the MLIP's predictions.
5.  **Compare and Visualise:** The core of this UAT is the comparison. The notebook will display a clean, formatted table:
    | Property | DFT Ground Truth | MLIP Prediction | Difference |
    |----------|------------------|-----------------|------------|
    | Energy   | -100.5 eV        | -100.4 eV       | 0.1 eV     |
    The notebook will also generate a scatter plot comparing the DFT forces to the MLIP forces for that structure. The points should lie very close to the y=x line, providing immediate visual confirmation of the model's accuracy.
6.  **Explanation:** A markdown cell will conclude: "The comparison above shows that the energy predicted by our newly trained potential is very close to the original DFT value. The force plot demonstrates that the potential has successfully learned to reproduce the atomic forces from the training data. The model is now ready for use in simulations."

---

## 2. Behavior Definitions

This section defines the expected behaviors of the system in the Gherkin-style Given/When/Then format.

### **UAT-C3-001: Successful End-to-End Training Run**

```gherkin
Feature: MLIP Model Training
  As a materials scientist,
  I want to train an MLIP from a dataset of DFT calculations,
  So that I can create a fast and accurate potential for simulations.

  Scenario: A valid dataset is used to successfully train a potential
    Given a database containing at least 20 valid DFT-calculated structures with energies and forces
    And a correctly configured Training Engine pointing to the database and a valid Pacemaker executable
    When I execute the training process
    Then the process should complete successfully without errors
    And the system should report the file path of a newly created potential file
    And the potential file should have a `.yace` extension.
```

### **UAT-C3-002: Verification of a Trained Potential**

```gherkin
Feature: Trained Potential Validation
  As a researcher,
  I want to verify that a trained potential is accurate,
  So that I can trust it for further scientific simulations.

  Scenario: Use a trained potential to predict forces and compare to ground truth
    Given a potential file that was successfully trained on a specific dataset
    And one atomic structure from that original dataset with its known DFT forces
    When I load the potential and use it to calculate the forces for that structure
    Then the calculated forces from the potential should closely match the known DFT forces
    And the Root Mean Squared Error (RMSE) between the potential's forces and the DFT forces should be below a small threshold (e.g., 0.1 eV/Angstrom).
```

### **UAT-C3-003: Handling of Training Process Failure**

```gherkin
Feature: Robustness in the Training Process
  As a user of an automated system,
  I want the system to handle failures gracefully and provide clear feedback,
  So that I can diagnose problems effectively.

  Scenario: The external training executable fails with an error
    Given a dataset that is intentionally corrupted (e.g., missing forces on some structures) which is known to cause the Pacemaker code to fail
    And a correctly configured Training Engine
    When I execute the training process
    Then the training process should fail
    And the system should not crash, but should instead raise a specific, informative Python exception (e.g., `TrainingError`)
    And the exception message should contain the captured error log from the failed external process, allowing for diagnosis.
```
