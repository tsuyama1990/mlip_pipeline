# Cycle 06 Specification: The Training Engine (Module D)

## 1. Summary

Cycle 06 deals with the "Learning" part of Machine Learning. We assume that the database is now populated with high-quality DFT data (thanks to Cycles 01-05). Now, we must map this data to a functional form: the Atomic Cluster Expansion (ACE) potential.

We integrate **Pacemaker**, a robust tool for training ACE potentials. The system must:
1.  **Harvest Data**: Query the database for `COMPLETED` structures and export them to the `.extxyz` format required by Pacemaker, splitting them into training and validation sets.
2.  **Configure Training**: Generate the complex `input.yaml` required by Pacemaker, setting hyperparameters like cutoff radius, basis size, and loss weights (Delta Learning).
3.  **Execute & Monitor**: Run the training process, capture the logs, and parse the resulting metrics (RMSE for Energy and Forces).
4.  **Versioning**: Save the resulting potential file (`output.yace`) with a version tag, allowing us to track the evolution of the potential over generations.

## 2. System Architecture

Files marked in **bold** are new or modified in this cycle.

### 2.1. File Structure

```ascii
mlip_autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── config/
│       │   └── schemas/
│       │       └── **training.py**     # Training Configuration Schema
│       ├── **training/**
│       │   ├── **__init__.py**
│       │   ├── **dataset.py**          # Data Export Logic
│       │   ├── **pacemaker.py**        # Wrapper for Pacemaker execution
│       │   └── **metrics.py**          # Log parsing & RMSE extraction
│       └── utils/
│           └── **io.py**               # Helper for atomic file IO
└── tests/
    └── training/
        ├── **test_dataset.py**
        └── **test_pacemaker.py**
```

### 2.2. Code Blueprints

#### `src/mlip_autopipec/config/schemas/training.py`
Defines the hyperparams.

```python
from pydantic import BaseModel, Field

class TrainingConfig(BaseModel):
    cutoff: float = 5.0
    test_fraction: float = 0.1
    energy_weight: float = 1.0
    force_weight: float = 100.0
    stress_weight: float = 1.0
    max_iter: int = 1000
    ladder_step: list = [100, 0.1] # Decay details
```

#### `src/mlip_autopipec/training/dataset.py`
Exports data for training.

```python
from mlip_autopipec.data_models.manager import DatabaseManager
from ase.io import write

class DatasetBuilder:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def build_dataset(self, output_dir: str, split: float = 0.1):
        """
        1. Query status='completed'.
        2. Shuffle.
        3. Split into train.xyz and test.xyz.
        4. Write files ensuring specific format (energy/forces keys).
        """
        atoms_list = self.db.get_completed_atoms()
        # shuffling and splitting logic
        write(f"{output_dir}/train.xyz", train_set, format='extxyz')
        write(f"{output_dir}/test.xyz", test_set, format='extxyz')
```

#### `src/mlip_autopipec/training/pacemaker.py`
Runs the trainer.

```python
import subprocess
from pathlib import Path

class PacemakerWrapper:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def train(self, dataset_dir: str, output_dir: str):
        """
        1. Generate input.yaml for pacemaker.
        2. Run `pacemaker input.yaml`.
        3. Capture stdout to `train.log`.
        """
        self._write_input_yaml(dataset_dir, output_dir)
        subprocess.run(["pacemaker", "input.yaml"], check=True)

    def _write_input_yaml(self, dataset_dir, output_dir):
        # Construct YAML structure matching Pacemaker docs
        pass
```

## 3. Design Architecture

### 3.1. Domain Concepts

1.  **Delta Learning**: To improve stability, we often fit $V_{total} = V_{ZBL} + V_{ACE}$. The `TrainingConfig` should support enabling/disabling the ZBL baseline.
2.  **Fitting Weights**: Forces are typically weighted $10\times$ to $100\times$ higher than energy because there are $3N$ force components per 1 energy value, and forces determine the dynamics.
3.  **Active Learning Iterations**: Each training run produces a potential "Generation N". We must store these distinct artifacts (e.g., `potentials/gen_01.yace`, `potentials/gen_02.yace`) to prevent overwriting.

### 3.2. Consumers and Producers

-   **Consumer**: `DatasetBuilder` consumes `COMPLETED` atoms from DB.
-   **Producer**: `PacemakerWrapper` produces a `.yace` file (the potential) and a report (metrics).

## 4. Implementation Approach

### Step 1: Data Export
-   **Task**: Implement `DatasetBuilder`.
-   **Detail**: Crucial point—Pacemaker expects specific keywords in the `.extxyz` file (e.g., `energy=...`, `forces=...`). ASE's default writer might put them in `info` or `arrays`. We must ensure mapping is correct so Pacemaker finds the labels.

### Step 2: Config Generation
-   **Task**: Implement `_write_input_yaml`.
-   **Detail**: This involves translating our Pydantic `TrainingConfig` into the specific YAML hierarchy Pacemaker uses (`cutoff`, `b_basis`, `loss`, etc.).

### Step 3: Execution Wrapper
-   **Task**: Implement `PacemakerWrapper`.
-   **Detail**: Use `subprocess`. Ensure we capture `stdout` to parse the RMSE later.

### Step 4: Metric Parsing
-   **Task**: Implement `LogParser`.
-   **Detail**: Regex search the log file for `RMSE_E` and `RMSE_F` values at the final epoch.

## 5. Test Strategy

### 5.1. Unit Testing Approach (Min 300 words)

-   **YAML Generation**:
    -   *Test*: Configure `TrainingConfig` with `cutoff=6.0`.
    -   *Action*: Call `_write_input_yaml`.
    -   *Assert*: The generated file contains `cutoff: 6.0`.
-   **Data Splitting**:
    -   *Test*: Mock DB returns 100 atoms. `split=0.1`.
    -   *Action*: Run `build_dataset`.
    -   *Assert*: `train.xyz` has 90 atoms, `test.xyz` has 10 atoms.
    -   *Assert*: Intersection is empty (no data leakage).
-   **Metric Parsing**:
    -   *Test*: Feed a sample Pacemaker log file.
    -   *Assert*: Returns correct RMSE values.

### 5.2. Integration Testing Approach (Min 300 words)

-   **Mock Training Run**:
    -   Since running actual training takes minutes/hours, we verify the *setup*.
    -   *Test*: Run `PacemakerWrapper.train()` but mock the `subprocess.run` to simple `touch output.yace`.
    -   *Assert*: Input files were created, subprocess was called with correct args, and the wrapper detected the "successful" completion (by checking file existence).
-   **End-to-End Data Flow**:
    -   Insert 5 atoms into DB.
    -   Run `DatasetBuilder`.
    -   Verify the `.extxyz` file is readable by ASE and contains the expected energy/forces fields.
