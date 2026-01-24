# Cycle 04 Specification: Training Orchestration

## 1. Summary
Cycle 04 implements the **Trainer**, which wraps the **Pacemaker** engine. This cycle is responsible for converting labeled data into Pacemaker-compatible formats, managing the **Delta Learning** setup (defining the physical baseline), and executing the training process. It also includes interfaces for **Active Set** selection to optimize data efficiency.

## 2. System Architecture

```ascii
mlip_autopipec/
├── config/
│   └── schemas/
│       └── training.py         # **Training Config Schema**
├── training/
│   ├── __init__.py
│   ├── pacemaker.py            # **Wrapper for pace_train/pace_activeset**
│   ├── dataset.py              # **Data Conversion (ASE -> ExtXYZ)**
│   └── metrics.py              # **Log Parsing (RMSE, etc.)**
```

## 3. Design Architecture

### 3.1. Pacemaker Wrapper (`training/pacemaker.py`)
A class to manage subprocess calls to `pace_train` and related tools.
- `train(dataset_path, initial_potential=None) -> potential_path`
- `select_active_set(candidates, current_potential) -> selected_indices`
- **Config Generation**: Automatically writes `input.yaml` for Pacemaker based on `TrainingConfig`.

### 3.2. Dataset Builder (`training/dataset.py`)
- **Format**: Pacemaker uses pickled dataframe or ExtXYZ. We will use **ExtXYZ** standard.
- **Splitting**: Logic to split data into `train.xyz` and `test.xyz` (e.g., 90/10 split).
- **Delta Learning**: If enabled, this module might need to subtract baseline energy/forces, OR configure Pacemaker to do it internally (Pacemaker supports `calc_driver` for baselines). *Design Decision: Use Pacemaker's internal reference potential feature.*

### 3.3. Training Config (`config/schemas/training.py`)
- `cutoff`: float
- `b_basis_size`: int
- `max_num_epochs`: int
- `batch_size`: int
- `ladder_step`: list (for hierarchical training)

## 4. Implementation Approach

1.  **Config**: Define `TrainingConfig`.
2.  **Dataset**: Implement `export_to_extxyz(atoms_list, filename)`. Ensure `energy` and `forces` properties are correctly mapped to ASE standard.
3.  **Wrapper**:
    - `write_input_yaml()`: Generate the YAML config for Pacemaker.
    - `run_training()`: Execute `pace_train`. Capture stdout/stderr.
    - `parse_metrics()`: Regex parse the logs to track RMSE evolution.
4.  **Active Set**: Implement wrapper for `pace_activeset`.

## 5. Test Strategy

### 5.1. Unit Testing
- **Config Gen**: Verify `input.yaml` matches Pacemaker's expected schema.
- **Dataset**: create random Atoms, export, read back, and verify properties match.
- **Log Parsing**: Feed sample Pacemaker logs and verify extracted RMSE values.

### 5.2. Integration Testing
- **Mock Pacemaker**: Create a shell script `pace_train` that acts as a dummy (just writes an output file).
- **Pipeline**:
    1.  Generate dummy ExtXYZ.
    2.  Call `PacemakerWrapper.train()`.
    3.  Verify it calls the binary with correct arguments and returns the path to the result.
