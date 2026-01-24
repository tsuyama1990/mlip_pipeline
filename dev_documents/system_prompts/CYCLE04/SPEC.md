# Cycle 04 Specification: Automated DFT Factory - Basic (Module C Part 1)

## 1. Summary

Cycle 04 marks the transition from "hypothetical" structures to "ground truth" data. We begin the implementation of **Module C: The Automated DFT Factory**. In this cycle, we focus on the "Happy Path": successfully generating Quantum Espresso input files, executing the binary via MPI, and parsing the resulting energies, forces, and stresses.

The challenge here is to automate the myriad of small decisions a human makes when running DFT:
-   How many k-points? (We will implement a density-based heuristic).
-   Which pseudopotentials? (We will integrate with the SSSP library).
-   What flags are needed for MLIP? (We strictly enforce `tprnfor=true` and `tstress=true`).

By the end of Cycle 04, the system will be able to take a `PENDING` structure from the database, run a static SCF calculation, and update the database with the calculated properties, setting the status to `COMPLETED`. We will *not* handle crashes or convergence failures in this cycle (that is Cycle 05).

## 2. System Architecture

Files marked in **bold** are new or modified in this cycle.

### 2.1. File Structure

```ascii
mlip_autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── dft/
│       │   ├── **__init__.py**
│       │   ├── **inputs.py**           # Input File Generation logic
│       │   ├── **runner.py**           # Subprocess Execution
│       │   └── **parsers.py**          # Output Parsing (XML/Text)
│       └── config/
│           └── schemas/
│               └── dft.py              # Already exists, might need updates
└── tests/
    └── dft/
        ├── **test_inputs.py**
        ├── **test_runner.py**
        └── **test_parsers.py**
```

### 2.2. Code Blueprints

#### `src/mlip_autopipec/dft/inputs.py`
Generates the text content for `pw.x` input.

```python
from ase import Atoms
from mlip_autopipec.config.schemas.dft import DFTConfig
import math

class InputGenerator:
    def __init__(self, config: DFTConfig):
        self.config = config

    def generate_input_string(self, atoms: Atoms) -> str:
        """
        Constructs the QE input file.
        - Calculates K-points from kspacing.
        - Assigns pseudopotentials from directory.
        - Sets &control, &system, &electrons.
        """
        kgrid = self._calculate_kgrid(atoms)
        # Logic to format the string...
        return "..."

    def _calculate_kgrid(self, atoms: Atoms) -> tuple:
        """k = ceil(2 * pi / (cell_length * kspacing))"""
        # ... implementation
        pass
```

#### `src/mlip_autopipec/dft/runner.py`
Executes the command.

```python
import subprocess
from pathlib import Path
from mlip_autopipec.data_models.manager import DatabaseManager
from mlip_autopipec.data_models.status import JobStatus

class QERunner:
    def __init__(self, config: DFTConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager

    def run_job(self, row_id: int):
        """
        1. Fetch atoms from DB.
        2. Generate input file.
        3. Run subprocess (mpirun ... pw.x).
        4. Parse output.
        5. Save results to DB.
        """
        atoms = self.db_manager.get_atom(row_id)
        # Setup run directory
        # Write input
        # Execute
        # Parse
        # Update DB
        pass
```

#### `src/mlip_autopipec/dft/parsers.py`
Parses the standard output or XML.

```python
import numpy as np

class QEOutputParser:
    def parse_output(self, stdout: str) -> dict:
        """
        Extracts:
        - Energy (eV)
        - Forces (eV/A) - (N, 3) array
        - Stress (eV/A^3) - (3, 3) array
        - Convergence status
        """
        # Regex or string search implementation
        return {
            "energy": ...,
            "forces": ...,
            "stress": ...
        }
```

## 3. Design Architecture

### 3.1. Domain Concepts

1.  **The "Static" Calculation**: For MLIP training, we almost *never* want to relax the structure (`calculation='relax'`). We want the forces on the *distorted* structure. Therefore, the runner strictly enforces `calculation='scf'`.
2.  **K-Point Density**: Instead of asking the user for "4x4x4", which is invalid if the supercell size changes, we ask for `kspacing` (inverse density, e.g., 0.15 $1/\text{\AA}$). This ensures consistent accuracy across different cell sizes (primitive vs supercell).
3.  **Pseudo-potential Management**: The system expects a directory (SSSP) where filenames follow a convention (e.g., `Fe.pbe-n-kjpaw_psl.1.0.0.UPF`). The `InputGenerator` maps element symbols to these files automatically.

### 3.2. Consumers and Producers

-   **Consumer**: `QERunner` consumes `PENDING` rows from the database.
-   **Producer**: `QERunner` updates the same rows, adding `data`, `forces`, `stress`, and changing status to `COMPLETED`.

## 4. Implementation Approach

### Step 1: Input Generation
We start here because it's purely textual and easy to test.
-   **Task**: Implement `InputGenerator`.
-   **Detail**: Ensure correct handling of `ntyp` (number of types) vs `nat` (number of atoms). Ensure magnetic moments are initialized if `nspin=2` is set in config.

### Step 2: Output Parsing
We need to extract data reliably.
-   **Task**: Implement `QEOutputParser`.
-   **Detail**: We can use `ase.io.read(..., format='espresso-out')` as a fallback, but a custom lightweight parser is often faster and less brittle for specific properties like "Job Done" verification.

### Step 3: The Runner Skeleton
-   **Task**: Implement `QERunner`.
-   **Detail**: It should create a temporary working directory (e.g., `_work/job_{id}/`), run the calculation, and then clean up (delete huge `.wfc` files) to save disk space.

### Step 4: Database Integration
-   **Task**: Update `DatabaseManager` (if needed) to support storing arrays (forces, stress). ASE db supports this natively via `data={...}`.

## 5. Test Strategy

### 5.1. Unit Testing Approach (Min 300 words)

-   **Input Generation**:
    -   *Test*: Pass an NaCl crystal. Check that the generated string contains `ATOMIC_SPECIES` with Na and Cl. Check that `K_POINTS` is automatic.
    -   *Test*: Pass a ferromagnetic Iron crystal. Check that `nspin=2` and `starting_magnetization` are present.
-   **Parsing**:
    -   *Test*: Feed a sample `pw.x` output file (stored in `tests/data/`).
    -   *Assert*: The parser extracts the correct total energy to 6 decimal places.
    -   *Assert*: The parser extracts forces shape (N, 3).
    -   *Assert*: The parser detects "JOB DONE" and returns success=True.
    -   *Test*: Feed a crashed output. Assert parser returns success=False.

### 5.2. Integration Testing Approach (Min 300 words)

-   **Mocked Binary Execution**:
    -   We cannot assume `pw.x` is installed on the CI runner. We will write a small shell script `mock_pw.sh` that simply `cat`s a pre-computed output file to stdout.
    -   *Test*: Configure `QERunner` to use `mock_pw.sh` as the command.
    -   *Action*: Run `run_job(id)`.
    -   *Expectation*: The runner creates the directory, "executes" the script, parses the "output", and updates the database status to `COMPLETED`.
-   **File System Cleanup**:
    -   *Test*: Run a job. Check that the temporary directory is removed (or kept if configured for debugging).
    -   *Test*: Check that large files (wavefunctions) are not left behind.
