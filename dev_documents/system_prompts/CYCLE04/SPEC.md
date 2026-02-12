# Cycle 04 Specification: Trainer (Pacemaker Integration)

## 1. Summary
**Goal**: Implement the `Trainer` component, the interface to the Pacemaker engine. This cycle enables the system to learn from data (ACE Potentials) and select the most informative data (Active Set Optimization / D-Optimality).

**Key Features**:
*   **Pacemaker Wrapper**: CLI execution of `pace_train` and `pace_activeset`.
*   **Delta Learning**: Incorporate LJ/ZBL baselines to ensure physical robustness.
*   **Active Set Optimization**: Use D-Optimality (MaxVol) to select a small subset of structures for DFT, drastically reducing cost.
*   **Dataset Management**: Handle efficient storage and merging of large atomic datasets (`.pckl.gzip`).

## 2. System Architecture

Files to be implemented/modified are **bold**.

```ascii
src/mlip_autopipec/
├── domain_models/
│   ├── **trainer.py**          # Training Config & Metrics
│   └── ...
├── trainer/
│   ├── **__init__.py**
│   ├── **base.py**             # Abstract Base Class
│   ├── **active_set.py**       # D-Optimality Selection
│   ├── **dataset.py**          # Dataset File Handling
│   ├── **pacemaker_wrapper.py**# CLI Wrapper
│   └── **delta.py**            # Baseline Potential Logic (LJ/ZBL)
└── tests/
    └── **test_trainer/**
        ├── **test_active_set.py**
        └── **test_dataset.py**
```

## 3. Design Architecture

### 3.1. Domain Models (`src/mlip_autopipec/domain_models/trainer.py`)

*   **`TrainerConfig`**:
    *   `r_cut`: Cutoff radius (float).
    *   `max_deg`: Maximum polynomial degree (int).
    *   `elements`: List[str] (e.g., `["Mg", "O"]`).
    *   `active_set_size`: Max number of structures to select (int).
*   **`TrainingMetrics`**:
    *   `rmse_energy` (float), `rmse_forces` (float).
    *   `validation_loss` (float).
    *   `training_time` (float).

### 3.2. Trainer Component (`src/mlip_autopipec/trainer/`)

#### `base.py`
*   **`BaseTrainer`** (ABC):
    *   `train(dataset: Path, initial_potential: Optional[Path]) -> PotentialArtifact`
    *   `select_active_set(candidates: List[Structure], current_potential: Optional[Path]) -> List[Structure]`

#### `dataset.py`
*   **`DatasetManager`**:
    *   **Responsibility**: Merge new DFT results into the main `accumulated.pckl.gzip`.
    *   **Efficiency**: Use streaming I/O or append mode if possible (Pacemaker usually requires rewriting the full pckl).
    *   **Validation**: Ensure atoms have `energy`, `forces`, and `stress` before adding.

#### `active_set.py`
*   **`ActiveSetSelector`**:
    *   Wraps `pace_activeset` CLI.
    *   **Logic**:
        *   Takes a large pool of `candidates`.
        *   Computes the determinant of the descriptor matrix (Information Matrix).
        *   Selects top `N` structures that maximize this determinant (D-Optimality).
    *   **Output**: A filtered list of `Structure` objects to be sent to Oracle.

#### `pacemaker_wrapper.py`
*   **`PacemakerTrainer`**:
    *   Wraps `pace_train` CLI.
    *   **Arguments**:
        *   `--dataset`: Path to pckl.gzip.
        *   `--initial_potential`: For fine-tuning (transfer learning).
        *   `--max_num_epochs`: Adjustable (e.g., 100 for initial, 10 for active learning updates).
    *   **Output**: `potential.yace` file.

#### `delta.py`
*   **`DeltaBaseline`**:
    *   Generates `input.yaml` for Pacemaker specifying a reference potential.
    *   **Strategy**:
        *   `ZBL` for short-range repulsion (mandatory).
        *   `LJ` for long-range van der Waals (optional).
    *   This ensures the ML part learns only the *difference* ($E_{DFT} - E_{Base}$).

## 4. Implementation Approach

1.  **Define Interfaces**: Create `base.py`.
2.  **Implement Dataset Logic**: Robust reading/writing of pickle files compatible with `ase`.
3.  **Implement Delta Logic**: Create simple ZBL parameters based on atomic numbers.
4.  **Implement CLI Wrappers**: Use `subprocess.run` to call `pace_train` and `pace_activeset`.
5.  **Mock Pacemaker**: Since Pacemaker might not be installed in all CI environments, create a mock that writes a dummy `.yace` file.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_dataset.py`**:
    *   Create dummy `ase.Atoms` with energy/forces.
    *   Save to `.pckl.gzip`.
    *   Read back and verify data integrity.
*   **`test_active_set.py`**:
    *   Mock `pace_activeset` call.
    *   Verify input arguments (cutoff, elements) are correctly passed.

### 5.2. Integration Testing
*   **Mock Training Loop**:
    *   Create a dataset.
    *   Run `PacemakerTrainer.train()` (mocked).
    *   Assert output file exists and metadata is returned.
