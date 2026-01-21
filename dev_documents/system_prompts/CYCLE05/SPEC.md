# Cycle 05 Specification: Active Learning Trainer

## 1. Summary

Cycle 05 implements **Module D: Training**. At this stage, the database is populated with "Labeled Data" (Atomic structures with corresponding DFT energies, forces, and stresses). The goal is to train a Machine Learning Interatomic Potential (MLIP) to reproduce these DFT results.

We use **Pacemaker** (a library for training Atomic Cluster Expansion or ACE potentials). This cycle bridges the gap between the raw data storage (ASE-db) and the external training engine. It involves:
1.  **Data Serialization**: Converting the SQL-based data into the `.pcfg` (Pickle Config) or `.extxyz` format required by Pacemaker. This includes splitting data into Training (90%) and Validation (10%) sets to monitor overfitting.
2.  **Hyperparameter Configuration**: Automatically generating the `input.yaml` file for Pacemaker. This involves setting physics-based constraints like cutoff radii and loss function weights. Typically, forces are weighted 100x more than energy because force vectors provide $3N$ data points per structure, whereas energy provides only 1.
3.  **Execution & Monitoring**: Running the training process, monitoring the RMSE (Root Mean Square Error) evolution in real-time (parsing logs), and saving the final `.yace` (YAML ACE) potential file.

This `.yace` file is the "Product" of the factory. It allows us to run simulations millions of times faster than DFT.

## 2. System Architecture

### File Structure
**bold** files are to be created or modified.

```
mlip_autopipec/
├── training/
│   ├── **__init__.py**
│   ├── **pacemaker.py**        # Wrapper for Pacemaker Execution
│   ├── **dataset.py**          # Data Export & Splitting Logic
│   └── **metrics.py**          # Log Parsing & RMSE Analysis
├── config/
│   └── schemas/
│       └── **training.py**     # Training hyperparams (cutoff, order)
```

### Data Dictionary

| Model Name | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| **TrainingConfig** | cutoff | float | Radial cutoff for the potential (Angstrom). |
| | b_basis_size | int | Number of basis functions (complexity). |
| | kappa | float | Weight for Energy loss. |
| | kappa_f | float | Weight for Force loss. |
| | max_iter | int | Maximum training epochs. |
| **TrainingMetrics** | epoch | int | Current epoch. |
| | rmse_energy | float | Energy error (meV/atom). |
| | rmse_force | float | Force error (eV/A). |

### Component Interaction
-   **`PacemakerWrapper`** calls **`dataset.export_data`** to create files.
-   **`PacemakerWrapper`** generates `input.yaml`.
-   **`PacemakerWrapper`** executes `pacemaker input.yaml`.
-   **`metrics.LogParser`** reads stdout/log.txt during execution.

## 3. Design Architecture

### Data Export (`dataset.py`)
-   **Query**: Select all records where `status='completed'`. We can filter by `generation` if we only want to train on recent data (Incremental Learning) or all data (Full Retraining).
-   **Format**: Write to `train.xyz` and `test.xyz` using `ase.io.write(format='extxyz')`. Pacemaker reads this natively.
-   **Metadata**: We must ensure that the `energy` and `forces` arrays are correctly labeled in the `.xyz` file (e.g., `energy=energy`, `forces=forces`).
-   **Splitting**: Random `train_test_split`. We use a fixed seed for reproducibility.

### Pacemaker Wrapper (`PacemakerWrapper`)
-   **Config Generation**: Map `TrainingConfig` (Pydantic) to Pacemaker's YAML syntax.
    -   `cutoff`: 5.0 - 6.0 Angstrom.
    -   `b_basis_size`: Controls the complexity (number of parameters).
    -   `loss`: `kappa` (Energy weight) vs `kappa_f` (Force weight).
-   **Process**: Run `pacemaker input.yaml`.
-   **Artifacts**: The run produces a `potential.yace` file. This must be versioned (e.g., `potential_gen_01.yace`) and stored in a consolidated `potentials/` directory.

### Metrics & Logging
-   Pacemaker writes a `log.txt` with RMSE per epoch.
-   We parse this to generate a learning curve (Training vs Validation Error).
-   **Early Stopping**: Ideally, Pacemaker handles this, but we can also implement a check: if Validation RMSE rises while Training RMSE falls (Overfitting), we stop.

## 4. Implementation Approach

1.  **Implement Exporter**: In `dataset.py`, create `export_data(output_dir)`. Ensure `config_type` is written to `info` dict in atoms so we can track it.
2.  **Implement Config Builder**: Create a template or a dictionary builder for `input.yaml` in `pacemaker.py`.
3.  **Implement Runner**: `PacemakerWrapper.train()`.
    -   Step 1: Export data.
    -   Step 2: Write config.
    -   Step 3: Run subprocess (`pacemaker input.yaml > log.txt`).
    -   Step 4: Check if `potential.yace` exists.
    -   Step 5: Copy potential to `production_potentials/`.
4.  **CLI**: `mlip-auto train`.

## 5. Test Strategy

### Unit Testing
-   **Config Builder**:
    -   Generate a config with `cutoff=4.5`. Verify YAML output contains `cutoff: 4.5`.
    -   Verify loss weights are set correctly (e.g., 100:1).
-   **Splitter**:
    -   Create 100 dummy atoms. Call split. Verify 90 in train, 10 in test. Verify no overlap between sets.

### Integration Testing
-   **Mock Training**:
    -   Since Pacemaker takes minutes/hours, we verify the wrapper by mocking the binary.
    -   The mock should read the input YAML and write a dummy `output.yace` file.
    -   The test asserts that the wrapper successfully calls the binary and locates the output file.
-   **Real Training (Smoke Test)**:
    -   If Pacemaker is installed, run on a tiny dataset (5 atoms).
    -   Verify it runs for 1 epoch and produces a valid file.
