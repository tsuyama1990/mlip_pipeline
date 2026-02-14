# Cycle 02 Specification: Oracle & Data Management

## 1. Summary
Cycle 02 implements the "Oracle" component, responsible for generating ground-truth data using Density Functional Theory (DFT), and the "Dataset Manager" for handling training data. This cycle bridges the gap between structure generation and potential training.

The key features are:
1.  **Automated DFT Execution**: A robust wrapper around Quantum Espresso (via ASE) that automatically generates input files, handles execution, and parses results (Energy, Forces, Stress).
2.  **Self-Healing Logic**: An intelligent error recovery system that detects SCF convergence failures and automatically retries calculations with adjusted parameters (e.g., reduced mixing beta, increased smearing).
3.  **Dataset Management**: A specialized module to read and write dataset files in the `.pckl.gzip` format required by Pacemaker, ensuring efficient storage and retrieval of large atomic structures.

## 2. System Architecture

The file structure expands `src/pyacemaker/oracle`. **Bold files** are new or modified.

```text
src/
└── pyacemaker/
    ├── core/
    │   └── **config.py**       # Updated with detailed DFTConfig
    └── **oracle/**
        ├── **__init__.py**
        ├── **manager.py**      # Main entry point for DFT tasks
        ├── **calculator.py**   # ASE Calculator Factory & Error Handlers
        └── **dataset.py**      # Dataset I/O (Pickle/Gzip)
```

### File Details
-   `src/pyacemaker/oracle/manager.py`: The high-level interface `DFTManager`. It accepts a list of `ase.Atoms`, configures the calculator, runs them (potentially in parallel), and returns the labelled atoms.
-   `src/pyacemaker/oracle/calculator.py`: Contains the logic to instantiate `ase.calculators.espresso.Espresso`. It includes the "Self-Correction" loop for handling crashes.
-   `src/pyacemaker/oracle/dataset.py`: Implements `DatasetManager` to save/load lists of `ase.Atoms` to `data/interim.pckl.gzip`.
-   `src/pyacemaker/core/config.py`: Expanded to include `DFTConfig` fields like `command`, `pseudopotentials`, `kspacing`, `basis_set`.

## 3. Design Architecture

### 3.1. DFT Configuration (Pydantic)
```python
class DFTConfig(BaseModel):
    code: str = "quantum_espresso"
    command: str = "mpirun -np 4 pw.x"
    pseudopotentials: Dict[str, str]  # e.g., {"Fe": "Fe.pbe.UPF"}
    kspacing: float = 0.04
    smearing: float = 0.02
    max_retries: int = 3
```

### 3.2. Dataset Manager
The `DatasetManager` must handle the specific serialization format used by Pacemaker.
```python
import pickle
import gzip
from ase import Atoms

class DatasetManager:
    def load(self, path: Path) -> List[Atoms]:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, data: List[Atoms], path: Path):
        with gzip.open(path, 'wb') as f:
            pickle.dump(data, f)
```

### 3.3. Self-Healing Oracle
The `DFTManager` implements a retry loop.
```python
class DFTManager(BaseModule):
    def compute(self, structure: Atoms) -> Optional[Atoms]:
        for attempt in range(self.config.dft.max_retries):
            try:
                # Configure calculator with attempt-specific params
                calc = self._create_calculator(structure, attempt)
                structure.calc = calc
                structure.get_potential_energy()
                return structure
            except Exception as e:
                self.logger.warning(f"DFT failed (Attempt {attempt}): {e}")
                # Analyze error and adjust params for next attempt
        return None  # Failed after retries
```

## 4. Implementation Approach

### Step 1: Update Configuration
-   Modify `src/pyacemaker/core/config.py` to add `DFTConfig` and `DatasetConfig`.
-   Add validation logic (e.g., check if pseudopotential files exist).

### Step 2: Dataset Manager
-   Implement `src/pyacemaker/oracle/dataset.py`.
-   Ensure compatibility with `ase.Atoms`.

### Step 3: Calculator Factory
-   Implement `src/pyacemaker/oracle/calculator.py`.
-   Create a factory function `create_calculator(config, attempt)` that returns an ASE calculator.
-   Implement logic to loosen convergence criteria (mixing beta, diagonalization) based on `attempt` number.

### Step 4: Oracle Manager
-   Implement `src/pyacemaker/oracle/manager.py`.
-   Integrate the calculator factory and the retry loop.
-   Add methods to handle batches of structures (sequential execution for now, parallel later).

## 5. Test Strategy

### 5.1. Unit Testing
-   **Dataset I/O (`tests/oracle/test_dataset.py`)**:
    -   Create a dummy `ase.Atoms` object.
    -   Save it using `DatasetManager`.
    -   Load it back and assert equality.
-   **Calculator Logic (`tests/oracle/test_calculator.py`)**:
    -   Test that `create_calculator` returns an object with correct parameters (kpts, pseudos) derived from config.
    -   Verify that `attempt=1` produces different parameters than `attempt=0` (e.g., smaller mixing beta).

### 5.2. Integration Testing
-   **Mock DFT Execution (`tests/oracle/test_manager.py`)**:
    -   Use `unittest.mock` to patch `ase.calculators.espresso.Espresso.get_potential_energy`.
    -   Simulate a success case: Check if `DFTManager.compute` returns the structure with energy.
    -   Simulate a failure case: Raise `Exception` in the mock. Verify that `DFTManager` retries `max_retries` times and then returns `None`.
    -   **Important**: Do NOT run actual `pw.x` in CI/CD.
