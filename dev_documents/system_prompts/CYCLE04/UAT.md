# Cycle 04 User Acceptance Test (UAT)

## 1. Test Scenarios

### **UAT-04: Active Learning & Training Loop**
*   **Goal**: Ensure that the Trainer can process datasets, select the most informative structures (Active Set), and fit the ACE potential using Pacemaker.
*   **Priority**: High
*   **Success Criteria**:
    *   The Trainer correctly converts ASE Structures (list) to Pacemaker Dataset (`.pckl.gzip`).
    *   The Active Set Selection (`pace_activeset`) correctly reduces the dataset size while preserving critical structures.
    *   The Training process (`pace_train`) runs without errors and produces a `.yace` potential file.

## 2. Behavior Definitions (Gherkin)

### Scenario: Dataset Conversion
**GIVEN** a list of 100 `Structure` objects with valid energy/forces
**WHEN** the Trainer converts this list to a Pacemaker dataset
**THEN** a `.pckl.gzip` file should be created
**AND** the file size should be non-zero
**AND** attempting to read it back (using ASE or pace tool) should yield 100 structures

### Scenario: Active Set Selection
**GIVEN** a dataset of 1000 structures
**AND** a configuration `active_set_size: 100`
**WHEN** the `ActiveSetSelector` processes the dataset
**THEN** a new dataset file (e.g., `activeset.pckl.gzip`) should be created
**AND** the new dataset should contain exactly 100 structures
**AND** the selection should complete successfully (exit code 0)

### Scenario: Potential Training
**GIVEN** a valid dataset path and a `TrainerConfig`
**WHEN** the `PacemakerTrainer.train()` method is called
**THEN** the `pace_train` command should be executed with correct arguments
**AND** a unique output directory (e.g., `training_run_001/`) should be created
**AND** the directory should contain `output_potential.yace` and `metrics.json`
**AND** the method should return a valid `Potential` object pointing to the `.yace` file
