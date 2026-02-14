# Cycle 03 Specification: Trainer & Potential Generation

## 1. Summary
Cycle 03 integrates the Pacemaker engine to train Atomic Cluster Expansion (ACE) potentials. This module is responsible for the "Refinement" phase of the active learning loop.

Key features:
1.  **Pacemaker Integration**: A robust wrapper around `pace_train` and `pace_activeset` commands, managing their execution and I/O.
2.  **Delta Learning**: Automatic configuration of physics-based baselines (Lennard-Jones or ZBL) to ensure core repulsion. The Trainer configures Pacemaker to learn only the difference ($E_{ACE} = E_{DFT} - E_{Baseline}$).
3.  **Active Set Optimization**: Implementation of D-optimality (MaxVol) selection to filter redundant structures from the training set, ensuring data efficiency.
4.  **Potential Versioning**: Management of potential files (`.yace`), tracking their lineage (generation number, parent potential).

## 2. System Architecture

The file structure expands `src/pyacemaker/trainer`. **Bold files** are new.

```text
src/
└── pyacemaker/
    ├── core/
    │   └── **config.py**       # Updated with TrainerConfig
    └── **trainer/**
        ├── **__init__.py**
        ├── **wrapper.py**      # Low-level Pacemaker CLI Wrapper
        ├── **active_set.py**   # Active Set Selection Logic
        └── **manager.py**      # High-level Trainer Class
```

### File Details
-   `src/pyacemaker/trainer/wrapper.py`: Contains `PacemakerWrapper` class. It uses `subprocess` to call `pace_train`, `pace_activeset`, and `pace_collect`.
-   `src/pyacemaker/trainer/active_set.py`: Contains `ActiveSetSelector` class. It uses `pace_activeset` to select optimal structures from a larger candidate pool.
-   `src/pyacemaker/trainer/manager.py`: Contains `Trainer` class. It orchestrates the training workflow: Data preparation -> Active Set Selection -> Training -> Potential saving.
-   `src/pyacemaker/core/config.py`: Expanded to include `TrainerConfig` (cutoff, order, basis_size, etc.).

## 3. Design Architecture

### 3.1. Trainer Configuration
```python
class TrainerConfig(BaseModel):
    cutoff: float = 5.0
    order: int = 3  # Maximum correlation order
    basis_size: Tuple[int, int] = (15, 5) # (radial, angular)
    delta_learning: str = "zbl"  # or "lj" or "none"
    max_epochs: int = 500
    batch_size: int = 100
```

### 3.2. Pacemaker Wrapper
```python
class PacemakerWrapper:
    def train(self, dataset_path: Path, output_dir: Path, params: dict):
        cmd = [
            "pace_train",
            "--dataset", str(dataset_path),
            "--output_dir", str(output_dir),
            # ... verify and append other params
        ]
        subprocess.run(cmd, check=True, capture_output=True)
```

### 3.3. Active Set Selection
```python
class ActiveSetSelector:
    def select(self, candidates_path: Path, num_select: int) -> Path:
        """Runs pace_activeset to select optimal structures."""
        # ... logic to run command
        return selected_dataset_path
```

## 4. Implementation Approach

### Step 1: Update Configuration
-   Modify `src/pyacemaker/core/config.py` to add `TrainerConfig`.

### Step 2: Pacemaker Wrapper
-   Implement `src/pyacemaker/trainer/wrapper.py`.
-   Use `shlex.split` for safe command construction.
-   Implement error handling for non-zero exit codes.

### Step 3: Active Set Logic
-   Implement `src/pyacemaker/trainer/active_set.py`.
-   Provide methods to select `N` structures from a `.pckl.gzip` file.

### Step 4: Trainer Manager
-   Implement `src/pyacemaker/trainer/manager.py`.
-   Implement `train_generation(dataset, initial_potential=None)` method.
-   If `initial_potential` is provided, configure `pace_train` to perform fine-tuning (start from existing weights).
-   If `delta_learning` is enabled, generate the appropriate baseline potential file (e.g., `ZBL.yace` or `LJ.yace`) before training.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Wrapper Testing (`tests/trainer/test_wrapper.py`)**:
    -   Mock `subprocess.run`.
    -   Verify that `pace_train` command is constructed correctly with all arguments (cutoff, order, etc.).
    -   Verify that `initial_potential` argument is added only when provided.
-   **Active Set Testing (`tests/trainer/test_active_set.py`)**:
    -   Mock `subprocess.run`.
    -   Verify that `pace_activeset` is called with correct input/output paths.

### 5.2. Integration Testing
-   **End-to-End Trainer Flow (`tests/trainer/test_manager.py`)**:
    -   Mock `PacemakerWrapper` and `ActiveSetSelector`.
    -   Run `Trainer.train_generation()`.
    -   Assert that the sequence is correct:
        1.  (Optional) Select active set.
        2.  Generate Baseline potential.
        3.  Run training.
        4.  Return path to new potential.
