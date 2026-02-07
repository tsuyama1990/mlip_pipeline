# Cycle 03 UAT: Learning from Data

## 1. Test Scenarios

### 1.1. Scenario: Data Conversion Fidelity
**ID**: UAT-CY03-001
**Priority**: High
**Description**: Verify that the system correctly converts ASE structures into the Pacemaker training format without losing data.

**Steps:**
1.  **Setup**: Create a list of 5 `ase.Atoms` with random positions and specific "Target Energy" tags (e.g., -100.5 eV).
2.  **Execution**: Call `dataset_manager.save_dataset(structures, "train.pckl.gzip")`.
3.  **Verification**: Load `train.pckl.gzip` using `pandas`. Check that the `energy` column exactly matches the input values.

### 1.2. Scenario: Training a Potential (Mocked Binary)
**ID**: UAT-CY03-002
**Priority**: Medium
**Description**: Verify that the Trainer module successfully orchestrates the training process, generating configuration files and invoking the backend.

**Steps:**
1.  **Setup**: Provide a valid `train.pckl.gzip` file.
2.  **Execution**: Run the `PacemakerTrainer.train()` method with `dry_run=False` (but pointing to a mock `pace_train` script).
3.  **Observation**: The logs should show "Starting Pacemaker training...", "Process ID: 12345", and finally "Training completed.".
4.  **Verification**: The output directory should contain `potential.yace` and `training_report.json`.

### 1.3. Scenario: Active Set Selection (D-Optimality)
**ID**: UAT-CY03-003
**Priority**: High
**Description**: Demonstrate that the system can select a diverse subset of structures from a larger pool.

**Steps:**
1.  **Setup**: Create a pool of 100 structures: 90 are nearly identical (Group A), 10 are distinct (Group B).
2.  **Execution**: Call `active_set.select(pool, n=15)`.
3.  **Verification**: The selected subset should contain all 10 structures from Group B and only 5 from Group A. This proves the algorithm prefers diversity.

## 2. Behaviour Definitions

**Feature**: Model Training Pipeline

**Scenario**: Successful Training Run

**GIVEN** a training dataset `data.pckl.gzip` with 100 structures
**AND** a configuration specifying `elements: [Si]` and `cutoff: 5.0`
**WHEN** the Trainer is executed
**THEN** a `input.yaml` file should be generated containing `cutoff: 5.0`
**AND** the `pace_train` command should be executed
**AND** a `output_potential.yace` file should be produced
**AND** the system should update the global state with the path to the new potential

**Scenario**: Training Divergence Handling

**GIVEN** a dataset with corrupted forces (e.g., NaNs)
**WHEN** the Trainer executes and `pace_train` returns a non-zero exit code
**THEN** the system should raise a `TrainingError`
**AND** the logs should capture the stderr from the subprocess
**AND** the Orchestrator should abort the cycle (or trigger a fallback strategy)
