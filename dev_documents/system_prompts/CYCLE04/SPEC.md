# Cycle 04 Specification: Trainer (Pacemaker Integration)

## 1. Summary

In this cycle, we integrate the **Pacemaker** engine (ACE potentials) into the pipeline. The Trainer module is responsible for managing the training dataset, selecting the most informative structures using D-Optimality (Active Set Selection), and fitting the Atomic Cluster Expansion (ACE) potential.

This module acts as a bridge between the Python ecosystem (ASE/Pydantic) and the Pacemaker CLI tools (`pace_train`, `pace_activeset`, `pace_collect`).

## 2. System Architecture

The following file structure will be created. **Bold** files are the targets for this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── training/
│   │   ├── **pacemaker.py**        # Wrapper for pace_train
│   │   ├── **activeset.py**        # Wrapper for pace_activeset
│   │   └── **base.py**             # (Update ABC with active set logic)
│   └── **dataset.py**              # Dataset Conversion Helpers
tests/
└── **test_trainer.py**             # Tests for Pacemaker CLI Calls
```

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/config.py`)
*   **`TrainerConfig`**:
    *   `dataset_path`: Path (Location of `training.pckl.gzip`)
    *   `initial_potential`: Optional[Path] (For fine-tuning)
    *   `max_num_epochs`: int (Training duration)
    *   `active_set_size`: int (Target number of structures for training)
    *   `cutoff`: float (ACE cutoff radius)
    *   `elements`: List[str] (e.g., ["Mg", "O"])

### 3.2. Trainer Components (`components/training/`)
*   **`PacemakerTrainer`**:
    *   `train(dataset: Dataset, initial_potential: Path) -> Potential`:
        1.  **Format Data**: Convert `Dataset` (list of `Structure`) to Pacemaker's `pckl.gzip` format using `dataset.py`.
        2.  **Select Active Set**: Call `ActiveSetSelector.select()` if `dataset > active_set_size`.
        3.  **Run Training**: Construct `pace_train` command.
            *   `--dataset`: Path to optimized dataset.
            *   `--initial_potential`: Only if provided (fine-tuning).
            *   `--output_dir`: Unique training directory.
        4.  **Parse Result**: Read metrics (RMSE, training time) from `metrics.json` (or log file). Return `Potential` object pointing to the best `.yace` file.

*   **`ActiveSetSelector`**:
    *   `select(dataset_path, n_select) -> Path`:
        *   Wraps `pace_activeset` command.
        *   Uses MaxVol algorithm (D-optimality) to select `n_select` structures.
        *   Returns path to the new, smaller dataset file.

### 3.3. Dataset Helpers (`components/dataset.py`)
*   **`DatasetConverter`**:
    *   `to_pacemaker(structures: List[Structure], path: Path)`:
        *   Uses `ase.io.write` or specific Pacemaker utility to save structures.
        *   Ensures all necessary properties (energy, forces, stress) are correctly mapped.

## 4. Implementation Approach

1.  **Trainer Config**: Define `TrainerConfig` and update `config.py`.
2.  **Dataset Logic**: Implement `to_pacemaker` conversion. Verify that `energy`, `forces` keys are preserved.
3.  **Active Set**: Implement `ActiveSetSelector` using `subprocess.run`.
    *   Handle cases where `pace_activeset` is not found (Mock/Skip).
4.  **Pacemaker Wrapper**: Implement `PacemakerTrainer.train`.
    *   Construct CLI arguments dynamically.
    *   Use `shutil` to manage output directories.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_dataset.py`**:
    *   Create `Structure` objects with energy/forces.
    *   Convert to `.pckl.gzip`.
    *   Read back using `ase.io.read` (if possible) or check file existence and size.

### 5.2. Integration Testing (Mocked Binary)
*   **`test_trainer.py`**:
    *   **Mock `subprocess.run`**: Intercept calls to `pace_train` and `pace_activeset`.
    *   **Verify CLI Construction**: Check that correct flags (`--dataset`, `--initial_potential`) are passed.
    *   **Verify Output Parsing**: Mock the creation of `output_potential.yace` and `metrics.json`. Assert the Trainer returns a valid `Potential` object.
