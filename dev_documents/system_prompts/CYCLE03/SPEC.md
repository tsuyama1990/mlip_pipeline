# Cycle 03 Specification: Machine Learning Core (Pacemaker Integration)

## 1. Summary
Cycle 03 integrates the "Learner" module. We will wrap the `pacemaker` library to enable the training of Atomic Cluster Expansion (ACE) potentials. This cycle involves managing datasets (serialization, merging), implementing the Active Set selection (D-Optimality) to choose the most informative structures, and executing the actual fitting process via `pace_train`. The goal is to go from a list of labeled structures (from Cycle 02) to a deployable `.yace` potential file.

## 2. System Architecture

### File Structure
Files to be created/modified are marked in **bold**.

```
mlip-pipeline/
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   ├── **config.py**       # Add TrainerConfig (Pacemaker params)
│       │   └── **dataset.py**      # Dataset serialization logic
│       ├── infrastructure/
│       │   ├── trainer/
│       │   │   ├── **__init__.py**
│       │   │   └── **pacemaker.py** # PacemakerWrapper implementation
│       │   └── active_learning/
│       │       ├── **__init__.py**
│       │       └── **selection.py** # Active Set Selection (MaxVol)
└── tests/
    └── integration/
        └── **test_trainer_pipeline.py**
```

## 3. Design Architecture

### Domain Models (`domain_models/`)

-   **`TrainerConfig`**:
    -   `cutoff`: float (Default: 5.0 Å)
    -   `max_basis_size`: int (Default: 500)
    -   `elements`: List[str]
    -   `energy_weight`: float (Default: 100.0)
    -   `force_weight`: float (Default: 1.0)
    -   `stress_weight`: float (Default: 0.1)

-   **`Dataset`**:
    -   `structures`: List[Structure]
    -   `save(path: Path)`: Serializes to `.pckl.gzip` (compatible with Pacemaker).
    -   `load(path: Path)`: Deserializes.
    -   `merge(other: Dataset)`: Combines two datasets.

### Infrastructure (`infrastructure/`)

-   **`PacemakerWrapper` (implements `BaseTrainer`)**:
    -   `train(dataset: Dataset, initial_potential: Optional[Path]) -> Path`:
        -   Saves dataset to disk.
        -   Calls `pace_train` via `subprocess`.
        -   Parses stdout for loss values (RMSE).
        -   Returns path to the best `.yace` file.
    -   `select_active_set(candidates: List[Structure], current_potential: Path) -> List[Structure]`:
        -   Uses `pace_activeset` to calculate MaxVol.
        -   Returns the subset of structures that maximize information gain.

## 4. Implementation Approach

1.  **Dataset Logic**: Implement robust serialization in `domain_models/dataset.py`. Use `pickle` with `gzip` compression. Ensure compatibility with `ase.io.read/write` for interoperability.
2.  **Pacemaker CLI Wrapper**:
    -   Implement `PacemakerWrapper.train` to construct the CLI command list.
    -   Handle `stdout` streaming to log progress (Epoch 1, Epoch 2...).
    -   Implement error handling for `pace_train` failures (e.g., singular matrix).
3.  **Active Set Selection**:
    -   Implement `selection.py` to wrap `pace_activeset`. This is critical for data efficiency.
    -   Parse the output to identify which structures were selected.
4.  **Integration**: Update `Orchestrator` to call `trainer.train()` after the `oracle.compute()` step.

## 5. Test Strategy

### Unit Testing (`tests/unit/`)
-   **`test_dataset.py`**:
    -   Create a dataset with random structures.
    -   Save and load it.
    -   Assert deep equality of the structures (positions, species, energy).

### Integration Testing (`tests/integration/`)
-   **`test_trainer_pipeline.py`**:
    -   Mock the `pace_train` binary (if not available in CI) to just touch the output file.
    -   If available, run a tiny training job (5 structures, 1 epoch).
    -   Assert that `potential.yace` is created.
    -   Assert that the log file contains "Training finished successfully".
