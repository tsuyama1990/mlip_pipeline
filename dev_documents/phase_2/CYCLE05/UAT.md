# Cycle 05 UAT: Active Learning & Training

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-05-01** | High | **Dataset Export & Delta Learning** | Verify that the system can query the database, subtract ZBL baseline energies, and export a formatted dataset compatible with Pacemaker. The exported energies must match (DFT - ZBL). |
| **UAT-05-02** | High | **Pacemaker Configuration** | Verify that the system correctly translates high-level user settings (e.g., "Forces Weight: 100") into the complex YAML format required by Pacemaker. |
| **UAT-05-03** | Medium | **Training Execution & Monitoring** | Verify that the system can launch the training process, monitor its progress (via logs), and capture the final potential file and accuracy metrics. |
| **UAT-05-04** | High | **Masking Respect** | Verify that if atoms in the database are marked with a `force_mask`, the generated training file sets their weights to 0. |

### Recommended Demo
Create `demo_05_training.ipynb`.
1.  **Block 1**: Populate a temporary ASE-db with 50 synthetic structures (e.g., Lennard-Jones atoms).
2.  **Block 2**: Initialize `DatasetBuilder` and run `export_data()`. Show the ZBL subtraction happening (print "Original E vs Delta E").
3.  **Block 3**: Setup `TrainConfig`.
4.  **Block 4**: Run `PacemakerWrapper.train()` (in mock mode).
5.  **Block 5**: Display the "Learning Curve" parsed from the mock logs (RMSE vs Epoch).
6.  **Block 6**: Show the final `.yace` file path.

## 2. Behavior Definitions

### Scenario: Delta Learning Correctness
**GIVEN** two Hydrogen atoms at distance 0.5 Angstrom (very close).
**AND** the DFT energy is +200 eV.
**AND** the ZBL energy for H-H at 0.5A is +190 eV.
**WHEN** the dataset is exported with `enable_delta_learning=True`.
**THEN** the energy value stored in the training file for this structure should be +10 eV (200 - 190).
**THIS** ensures the ML model only learns the residual.

### Scenario: Config Generation
**GIVEN** a `TrainConfig` with `cutoff=5.5` and `loss_weights={'forces': 50.0}`.
**WHEN** the Pacemaker input file is generated.
**THEN** the file should contain `cutoff: 5.5`.
**AND** the file should contain `weight: 50.0` inside the force loss section.
**AND** the file should point to the correct `train.pckl.gzip` path.

### Scenario: Training Artifacts
**GIVEN** a successful training run.
**WHEN** the wrapper returns.
**THEN** a file named `output.yace` (or similar) should exist in the run directory.
**AND** a log file should exist.
**AND** the result object should contain `rmse_forces` parsed from the log.

### Scenario: Force Masking
**GIVEN** a structure where atom 0 has `force_mask=1` and atom 1 has `force_mask=0`.
**WHEN** dataset is exported.
**THEN** the weight array for this structure should be `[1.0, 0.0]`.
**THIS** ensures we don't train on artifacts.
