# Cycle 04 Specification: Training & Active Learning

## 1. Summary

In Cycle 04, we integrate the **Pacemaker** engine to enable the actual "Learning" phase. We replace `MockTrainer` with a robust wrapper that manages the lifecycle of the ACE potential.

A key challenge in MLIP generation is the explosion of data. Simply feeding thousands of similar structures to the trainer leads to slow training and no gain in accuracy. We implement **Active Set Optimization** (using D-Optimality / MaxVol algorithm provided by `pace_activeset`). Before training, the system analyzes the geometric descriptors of the new candidate structures and selects only the most "novel" ones to add to the training set.

We also implement rigid **Data Management**. Training data is valuable. We must persist the accumulated structures (with their provenance tags) in a reliable format (Pickle/Gzip or ASE Database), ensuring we can restart or branch the project at any time.

## 2. System Architecture

```ascii
src/mlip_autopipec/
├── ...
├── training/
│   ├── __init__.py
│   ├── **pacemaker_wrapper.py** # CLI Driver for pace_train
│   ├── **active_set.py**        # Logic for pace_activeset
│   └── **dataset.py**           # Data persistence (.pckl.gzip)
└── ...
```

## 3. Design Architecture

### 3.1. Pacemaker Wrapper (`pacemaker_wrapper.py`)
*   **`PacemakerTrainer`**: Prepares the `input.yaml` for Pacemaker.
    *   **Delta Learning**: Automatically configures `bonds` section for ZBL/LJ baselines.
    *   **Execution**: Calls `pace_train` via `subprocess`.
    *   **Parsing**: Reads `log.txt` to extract RMSE and loss evolution.

### 3.2. Active Set Selection (`active_set.py`)
*   **`ActiveSetSelector`**:
    *   Input: `List[Structure]` (candidates), `List[Structure]` (existing set).
    *   Process: Runs `pace_activeset` to compute the information matrix determinant.
    *   Output: `List[Structure]` (reduced set).

### 3.3. Dataset Management (`dataset.py`)
*   **`DatasetManager`**:
    *   `load(path)` / `save(path)`
    *   `merge(new_data)`: Smart merging that prevents duplicates (based on structure hash).

## 4. Implementation Approach

1.  **Dataset Utils**: Implement `DatasetManager` to handle `ase.io.read/write` with `.pckl.gzip` compression (standard for Pacemaker).
2.  **Active Set Logic**: Implement the wrapper for `pace_activeset`. Note: This requires the `pacemaker` python environment.
3.  **Training Logic**: Implement `PacemakerTrainer.train()`.
    *   Generate `input.yaml` from Pydantic `TrainingConfig`.
    *   Run training.
    *   Copy the resulting `potential.yace` to the `potentials/` archive.
4.  **Orchestrator Update**: Inject the real `PacemakerTrainer`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Config Generation**: Create a `TrainingConfig` object and assert that the generated `input.yaml` string contains the correct `cutoff`, `bonds`, and `fitting` parameters.
*   **Dataset Merge**: Merge two lists of atoms with one duplicate. Verify the result length.

### 5.2. Integration Testing
*   **Full Train Cycle (Mocked Binary)**: If `pace_train` is not installed, mock the subprocess call. Verify that the wrapper correctly constructs the CLI arguments.
*   **Real Train Cycle**: If available, run a training on 5 atoms for 1 epoch. Verify a `.yace` file is produced.
