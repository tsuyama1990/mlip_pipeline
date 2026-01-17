# Cycle 05: User Acceptance Testing (UAT)

## 1. Test Scenarios

Cycle 05 is about learning. We verify the system can translate data into a model.

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-05-01** | High | Training Config Generation | Verify that the system generates a valid Pacemaker configuration file, including Delta Learning settings. |
| **UAT-05-02** | Medium | Metrics Extraction | Verify that the system correctly parses training logs to report Model Performance (RMSE). |

### Recommended Notebooks
*   `notebooks/UAT_05_Learning.ipynb`:
    1.  Define training hyperparameters (cutoff, order).
    2.  Generate the `input.yaml`.
    3.  Display the YAML content to verify ZBL settings are present.
    4.  Simulate a training run (copy a sample log file).
    5.  Parse and plot the learning curve (RMSE vs Epoch).

## 2. Behavior Definitions

### UAT-05-01: Config Generation

**Narrative**:
The user wants to train a potential for Fe-C. They know that short-range repulsion is critical. They rely on the system to set up the "Delta Learning" baseline. The Config Generator reads the atomic numbers (26 and 6), calculates the ZBL screening lengths, and writes the `input.yaml`. The user inspects the file and confirms that the `zbl` section is correctly populated with the specific parameters for Fe-Fe, Fe-C, and C-C.

```gherkin
Feature: Training Configuration

  Scenario: Generating Pacemaker Config with Delta Learning
    GIVEN a request to train a potential for Fe-Ni with a cutoff of 5.0 Angstroms
    WHEN the Pacemaker Config Generator is called
    THEN the resulting YAML file should have "cutoff: 5.0"
    AND it should include a "zbl" section for inner-core repulsion
    AND the weights for Energy, Force, and Stress should match the system defaults (1:100:10)
    AND the basis set size (max_degree) should be appropriate for the number of elements

  Scenario: Handling Missing Data
    GIVEN a database with only 2 structures
    WHEN the Config Generator is called
    THEN it should warn about overfitting
    OR automatically reduce the basis set size
```

### UAT-05-02: Performance Reporting

**Narrative**:
Training has finished. The user wants to know if the potential is good. They don't want to grep through a 10MB log file. The Metrics Parser reads the log, finds the final test set RMSE for forces, and reports "RMSE_F = 5.2 meV/A". It also checks if this meets the success criteria (< 10 meV/A).

```gherkin
Feature: Metrics Parsing

  Scenario: Parsing Training Logs
    GIVEN a Pacemaker log file indicating a final Force RMSE of 5 meV/A
    WHEN the Metrics Parser reads the log
    THEN it should return a dictionary {"force_rmse": 0.005, "is_converged": True}
    AND this metric should be logged to the system status
    AND if the RMSE was 50 meV/A, "is_converged" should be False
```
