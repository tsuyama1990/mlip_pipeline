# Cycle 04: User Acceptance Test (UAT) Plan

## 1. Test Scenarios

### Scenario 4.1: Dataset Creation & Update
**Priority**: High
**Description**: Verify that `CalculationResult` objects from the Oracle can be correctly converted into the format required by Pacemaker (Pickled Pandas DataFrame) and appended to a persistent dataset.

**Jupyter Notebook**: `tutorials/03_trainer_test.ipynb`
1.  Create 10 dummy `CalculationResult` objects with energies and forces (using `MockOracle` from Cycle 03).
2.  Initialize `PacemakerTrainer`.
3.  Call `trainer.update_dataset(results)`.
4.  Assert that a `data/accumulated.pckl.gzip` file is created.
5.  Load the file using `pandas.read_pickle`.
6.  Assert that the DataFrame has 10 rows and correct columns (`energy`, `forces`).
7.  Add 5 more results and call `update_dataset` again.
8.  Assert that the DataFrame now has 15 rows.

### Scenario 4.2: Active Set Selection (D-Optimality)
**Priority**: Medium
**Description**: Verify that the system can select the most informative structures from a candidate pool using `pace_activeset`.

**Jupyter Notebook**: `tutorials/03_trainer_test.ipynb`
1.  Generate a pool of 20 diverse structures (e.g., perturbed bulk).
2.  Mock the `pace_activeset` command (or use real if installed).
3.  Call `trainer.select_active_set(pool, n=5)`.
4.  Assert that the returned list contains exactly 5 structures.
5.  (Optional) Verify that the selected structures are distinct from each other.

### Scenario 4.3: Training Execution (Mock)
**Priority**: Critical
**Description**: Verify that the trainer can orchestrate the execution of `pace_train` and produce a potential artifact.

**Jupyter Notebook**: `tutorials/03_trainer_test.ipynb`
1.  Prepare a small dataset file.
2.  Mock `subprocess.run` to simulate `pace_train` success (create a dummy `.yace` file).
3.  Call `trainer.train(dataset_path)`.
4.  Assert that the function returns a valid `PotentialArtifact`.
5.  Assert that the artifact's path points to the dummy `.yace` file.
6.  Verify that the generated `input.yaml` contained the correct parameters from config.

### Scenario 4.4: Fine-Tuning Workflow
**Priority**: High
**Description**: Verify that the system supports fine-tuning an existing potential.

**Jupyter Notebook**: `tutorials/03_trainer_test.ipynb`
1.  Create a dummy `initial.yace`.
2.  Call `trainer.train(dataset_path, initial_potential="initial.yace")`.
3.  Inspect the generated `input.yaml` passed to the mock `pace_train`.
4.  Assert that it contains the line `initial_potential: initial.yace` (or equivalent flag).

## 2. Behavior Definitions

### Dataset Management
**GIVEN** a list of new calculation results
**WHEN** `update_dataset` is called
**THEN** the new data should be appended to the existing dataset on disk
**AND** duplicate structures (if any) should be handled (e.g., kept or ignored based on policy).

### Active Set Logic
**GIVEN** a large pool of candidate structures
**WHEN** `select_active_set` is invoked
**THEN** it should call the `pace_activeset` CLI tool with the correct arguments
**AND** parse the output to identify the indices of selected structures.

### Training Loop
**GIVEN** a valid configuration and dataset
**WHEN** `train` is executed
**THEN** it should run `pace_train` with the specified hyperparameters
**AND** return the path to the best-performing potential found during training.
