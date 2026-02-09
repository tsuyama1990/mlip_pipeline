# Cycle 04 Specification: Trainer (Pacemaker Integration)

## 1. Summary

Cycle 04 integrates the core machine learning engine: **Pacemaker (ACE: Atomic Cluster Expansion)**. The `Trainer` component takes the labeled data from the Oracle (Cycle 03) and produces a machine learning interatomic potential (`potential.yace`). This cycle also introduces "Active Set Selection" (D-Optimality) to improve data efficiency, and "Delta Learning" to enforce physical robustness.

Key features:
1.  **Pacemaker Wrapper**: Automate the execution of `pace_train` and `pace_activeset` via subprocess calls.
2.  **Dataset Management**: Collect labeled `ase.Atoms` objects and serialize them into the format Pacemaker expects (`dataset.pckl.gzip`).
3.  **Active Set Selection**: Before training, use `pace_activeset` to select only the most informative structures (maximizing the determinant of the descriptor matrix). This is crucial for reducing training time and avoiding redundancy.
4.  **Delta Learning Config**: Automatically generate the YAML configuration for Pacemaker to learn the difference between DFT and a physics baseline (LJ/ZBL).

By the end of this cycle, the Orchestrator will be able to take a list of labeled structures and produce a trained `potential.yace` file ready for use in simulations.

## 2. System Architecture

This cycle focuses on the `components/trainer` package.

### File Structure
Files to be created/modified in this cycle are marked in **bold**.

```
src/mlip_autopipec/
├── components/
│   ├── trainer/
│   │   ├── __init__.py
│   │   ├── **base.py**             # Enhanced Abstract Base Class
│   │   ├── **pacemaker_driver.py** # Wrapper for `pace_*` commands
│   │   ├── **active_set.py**       # Logic for D-optimality selection
│   │   └── **dataset.py**          # Data collection & serialization
│   └── factory.py                  # Update to register Trainer
├── domain_models/
│   └── **config.py**               # Add TrainerConfig details
└── tests/
    └── **test_trainer.py**
```

## 3. Design Architecture

### 3.1. Trainer Configuration (`domain_models/config.py`)
Update `TrainerConfig` to include:
*   `max_num_epochs`: int (e.g., 500).
*   `batch_size`: int (e.g., 10).
*   `energy_weight`: float (e.g., 100.0).
*   `force_weight`: float (e.g., 1.0).
*   `use_active_set`: bool.
*   `active_set_config`: Dict (selection method: MaxVol).
*   `physics_baseline`: Dict (e.g., `{"type": "lj", "sigma": 2.5, "epsilon": 0.1}`).

### 3.2. Pacemaker Driver (`components/trainer/pacemaker_driver.py`)
This class manages the interaction with the Pacemaker binary.
*   `prepare_dataset(structures)`: Converts a list of `Structure` objects to a `pandas.DataFrame` and saves as `dataset.pckl.gzip`.
*   `select_active_set(dataset_path)`: Calls `pace_activeset` to filter the dataset.
*   `train(dataset_path, output_dir)`:
    1.  Generate `input.yaml` for Pacemaker (incorporating baseline settings).
    2.  Run `pace_train`.
    3.  Monitor logs for convergence.
    4.  Return path to `potential.yace`.

### 3.3. Active Set Selector (`components/trainer/active_set.py`)
This module implements the logic to decide *which* structures enter the training set.
*   **MaxVol Algorithm**: Selects a subset of structures that maximizes the volume of the descriptor parallelepiped.
*   **Benefits**: Reduces overfitting and speeds up training by discarding redundant data.

## 4. Implementation Approach

1.  **Implement `Dataset` Manager**: Create a class to handle the accumulation of data across cycles. It must support appending new data to an existing dataset file.
2.  **Implement `PacemakerDriver`**:
    *   Use `subprocess.run` to call `pace_train`.
    *   Important: Ensure the `input.yaml` generation correctly specifies the `cutoff`, `b_basis`, and `functions`.
3.  **Implement Delta Learning Logic**:
    *   In `input.yaml`, add the `potential: delta` section if a baseline is configured.
    *   This tells Pacemaker to subtract the baseline energy before fitting.
4.  **Orchestrator Integration**: Add the Trainer step after Oracle.
    *   Take labeled data -> Update Dataset -> Select Active Set -> Train -> Save `potential.yace`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Dataset Conversion**:
    *   Input: List of 10 `Structure` objects.
    *   Action: `Dataset.create(structures)`.
    *   Assert: Returns a valid path to a `.pckl.gzip` file. The file can be loaded by `pandas`.
*   **Config Generation**:
    *   Input: `TrainerConfig` with `lj` baseline.
    *   Action: `driver._generate_yaml()`.
    *   Assert: The YAML string contains `potential: delta` and `base_potential: lj`.

### 5.2. Integration Testing
*   **Mock Training**:
    *   Mock `subprocess.run` for `pace_train`.
    *   Simulate the creation of a dummy `output_potential.yace` file.
    *   Run `trainer.train()`.
    *   Assert: The method returns the path to the potential file.
